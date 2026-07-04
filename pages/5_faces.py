"""Owner Faces page: detect → cluster → name → merge/move → apply, one batch at a time."""
import streamlit as st

from photopipe.config import get_config
from photopipe.database import Database
from photopipe.faces.service import FaceService
from photopipe.models import PhotoPhase

st.set_page_config(page_title="Faces - PhotoPipe", page_icon="🧑", layout="wide")


@st.cache_resource(show_spinner="Loading face model…")
def _face_backend():
    """One InsightFace model per server process — reloading the ~300 MB
    model pack on every Streamlit rerun costs seconds and ~1 GB of churn."""
    from photopipe.faces.detector import InsightFaceBackend

    return InsightFaceBackend()


def init_state() -> None:
    if "db" not in st.session_state:
        get_config().ensure_directories()
        st.session_state.db = Database()
    st.session_state.setdefault("faces_min_cluster_size", 3)


def _cluster_display(cluster, members) -> str:
    label = cluster.label or "unnamed"
    return f"{label} · {len(members)} faces"


def main() -> None:
    init_state()
    db = st.session_state.db

    st.title("🧑 Faces")
    st.caption(
        "Detect and group faces, then name each person once. "
        "All face data stays on this machine."
    )

    batches = [
        b for b in db.get_all_batches()
        if any(p.phase != PhotoPhase.FINALIZED for p in db.get_photos_by_batch(b.id))
    ]
    if not batches:
        st.info("No batches to work on. Convert a bucket on the Buckets page first.")
        return

    chosen = st.selectbox(
        "Batch",
        range(len(batches)),
        format_func=lambda i: batches[i].name,
        help="Faces are detected and grouped one batch at a time.",
    )
    batch = batches[chosen]

    min_size = st.slider(
        "Minimum faces per person", 2, 8,
        st.session_state.faces_min_cluster_size,
        help="A person needs at least this many face crops to form a group.",
    )
    st.session_state.faces_min_cluster_size = min_size
    svc = FaceService(db, backend=_face_backend(), min_cluster_size=min_size)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(
            "1 · Detect faces", use_container_width=True,
            help="Finds every face in the batch. The first run downloads the "
                 "face model (~300 MB); everything stays on this machine.",
        ):
            bar = st.progress(0.0, text="Detecting…")
            result = svc.detect_batch(
                batch, progress=lambda c, t: bar.progress(c / t, text=f"{c}/{t}")
            )
            st.success(
                f"Found {result.faces_found} faces in {result.photos_scanned} photos"
            )
            for err in result.errors:
                st.warning(err)

    with col2:
        if st.button(
            "2 · Group faces", use_container_width=True,
            help="Clusters the detected faces by person.",
        ):
            result = svc.cluster_batch(batch)
            st.success(
                f"{result.cluster_count} people, {result.noise_count} unsorted faces"
            )

    with col3:
        if st.button(
            "3 · Apply names", type="primary", use_container_width=True,
            help="Adds each person's name to the keywords of the photos they "
                 "appear in. Re-applying after a rename updates the keywords.",
        ):
            result = svc.propagate_labels(batch)
            st.success(
                f"Tagged {result.photos_tagged} photos with "
                f"{result.names_applied} names"
            )

    st.markdown("---")

    clusters = db.get_face_clusters_by_batch(batch.id)
    faces = db.get_faces_by_batch(batch.id)
    faces_by_cluster: dict[str, list] = {}
    for f in faces:
        faces_by_cluster.setdefault(f.cluster_id, []).append(f)

    if not clusters:
        st.info("Run Detect then Group to see people here.")
        return

    real = [c for c in clusters if not c.is_noise]
    noise = [c for c in clusters if c.is_noise]

    for cluster in real:
        members = faces_by_cluster.get(cluster.id, [])
        st.subheader(f"Person · {len(members)} faces")
        new_label = st.text_input(
            "Name", value=cluster.label or "", key=f"name_{cluster.id}",
            placeholder="e.g., Grandma Rose",
            help="This name is added to the photo keywords when you Apply names.",
        )
        # Compare stripped values: name_cluster stores the stripped label, so
        # comparing the raw widget value would rerun forever on "Rose ".
        if new_label.strip() != (cluster.label or ""):
            svc.name_cluster(cluster.id, new_label)
            st.rerun()
        cols = st.columns(10)
        for i, face in enumerate(members[:10]):
            with cols[i]:
                if face.crop_path and face.crop_path.exists():
                    st.image(str(face.crop_path))

    if len(real) >= 2:
        with st.expander("Merge groups (one person split into two groups)"):
            options = list(range(len(real)))
            picked = st.multiselect(
                "Groups to merge",
                options,
                format_func=lambda i: _cluster_display(
                    real[i], faces_by_cluster.get(real[i].id, [])
                ),
                help="The first selected group is kept; the others are folded "
                     "into it (a name on a merged group carries over if the "
                     "kept one is unnamed).",
            )
            if st.button("Merge selected", disabled=len(picked) < 2):
                svc.merge_clusters([real[i].id for i in picked])
                st.rerun()

    for cluster in noise:
        members = faces_by_cluster.get(cluster.id, [])
        if members:
            st.subheader(f"Unsorted faces · {len(members)}")
            st.caption("Faces that didn't group with anyone.")
            cols = st.columns(10)
            for i, face in enumerate(members[:10]):
                with cols[i]:
                    if face.crop_path and face.crop_path.exists():
                        st.image(str(face.crop_path))
            if real:
                with st.expander("Move an unsorted face into a group"):
                    for face in members[:10]:
                        fc1, fc2, fc3 = st.columns([1, 3, 1])
                        with fc1:
                            if face.crop_path and face.crop_path.exists():
                                st.image(str(face.crop_path))
                        with fc2:
                            target = st.selectbox(
                                "Group",
                                range(len(real)),
                                format_func=lambda i: _cluster_display(
                                    real[i], faces_by_cluster.get(real[i].id, [])
                                ),
                                key=f"move_target_{face.id}",
                                label_visibility="collapsed",
                            )
                        with fc3:
                            if st.button("Move", key=f"move_{face.id}"):
                                svc.move_face(face.id, real[target].id)
                                st.rerun()


if __name__ == "__main__":
    main()
