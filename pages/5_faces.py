"""Owner Faces page: detect → cluster → name → apply, one batch at a time."""
import streamlit as st

from photopipe.config import get_config
from photopipe.database import Database
from photopipe.faces.service import FaceService
from photopipe.models import PhotoPhase

st.set_page_config(page_title="Faces - PhotoPipe", page_icon="🧑", layout="wide")


def init_state() -> None:
    if "db" not in st.session_state:
        get_config().ensure_directories()
        st.session_state.db = Database()
    st.session_state.setdefault("faces_min_cluster_size", 3)


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

    chosen = st.selectbox("Batch", [b.name for b in batches])
    batch = next(b for b in batches if b.name == chosen)

    min_size = st.slider(
        "Minimum faces per person", 2, 8,
        st.session_state.faces_min_cluster_size,
        help="A person needs at least this many face crops to form a group.",
    )
    st.session_state.faces_min_cluster_size = min_size
    svc = FaceService(db, min_cluster_size=min_size)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("1 · Detect faces", use_container_width=True):
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
        if st.button("2 · Group faces", use_container_width=True):
            result = svc.cluster_batch(batch)
            st.success(
                f"{result.cluster_count} people, {result.noise_count} unsorted faces"
            )

    with col3:
        if st.button("3 · Apply names", type="primary", use_container_width=True):
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
        )
        if new_label != (cluster.label or ""):
            svc.name_cluster(cluster.id, new_label)
            st.rerun()
        cols = st.columns(10)
        for i, face in enumerate(members[:10]):
            with cols[i]:
                if face.crop_path and face.crop_path.exists():
                    st.image(str(face.crop_path))

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


if __name__ == "__main__":
    main()
