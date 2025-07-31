from PIL import Image
from worker import Worker
from draw_utils import draw_cluster, draw_contour, draw_rect
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

worker = Worker()

if 'images' not in st.session_state:
    st.session_state.images = []

if 'rect' not in st.session_state:
    st.session_state.rect = []

st.subheader("Algorith parameters:")

st.write(f'**Device**: {worker.device}')

st.subheader("Clustering")
for key, value in worker.config.cluster.items():
    st.write(f"**{key}:** {value}")

st.subheader("Contours")
for key, value in worker.config.contour.items():
    st.write(f"**{key}:** {value}")

uploaded_file = st.file_uploader("Load image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    original_img = Image.open(uploaded_file).resize((512,512))
    canvas_result = st_canvas(
        stroke_width=3,
        stroke_color='green',
        background_color="",
        background_image=original_img,
        update_streamlit=True,
        height=512,
        width=512,
        drawing_mode='point',
        point_display_radius=3,
        key="canvas",
    )
    
    if st.button("Clear"):
        st.session_state.images = []
        st.session_state.rect = []
        st.rerun()

    if st.button("VCD") and canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        
        if len(objects) == 2:
            rect_img = original_img.copy()
            draw_rect(rect_img, objects)

            mask_img = original_img.copy()
            mask_img = worker.get_binary_mask(np.asarray(mask_img))
            st.session_state.images.append({'caption':'segmentation', 'image':mask_img})

            countour_img = rect_img.copy()
            contour_points = worker.get_contour_points()
            draw_contour(countour_img, contour_points)

            cluster_img = countour_img.copy()
            cluster1, cluster2 = worker.get_clusters(st.session_state.rect)
            draw_cluster(cluster_img, cluster1, cluster2)
    
    for item in st.session_state.images:
        st.subheader(item['caption'])
        st.image(image=item['image'])