import streamlit as st
from typing import  List, Dict
import numpy as np
from PIL import Image, ImageDraw


def draw_rect(img: Image.Image, rect_points: List[Dict]):
    p1 = rect_points[0]
    p2 = rect_points[1]

    x1, y1 = p1["left"], p1["top"]
    x2, y2 = p2["left"], p2["top"]

    rect_left = min(x1, x2)
    rect_top = min(y1, y2)
    rect_width = abs(x1 - x2)
    rect_height = abs(y1 - y2)
    st.session_state.rect = [rect_left, rect_top, rect_width, rect_height]

    draw = ImageDraw.Draw(img)
    draw.rectangle(
        [(rect_left, rect_top), (rect_left + rect_width, rect_top + rect_height)],
        outline="green",
        width=2
    )

    st.session_state.images.append({'caption':'rectangle', 'image':img})


def draw_contour(img: Image.Image, points: np.ndarray):
    draw = ImageDraw.Draw(img)

    for y, x in points:
        draw.circle((x,y), radius=1,outline='red', fill='red')
    
    st.session_state.images.append({'caption': 'contours', 'image':img})


def draw_cluster(img: Image.Image, cluster1: np.ndarray, cluster2: np.ndarray):
    draw = ImageDraw.Draw(img)

    for y, x in cluster1:
        draw.circle((x,y), radius=1,outline='yellow', fill='yellow')

    for y, x in cluster2:
        draw.circle((x,y), radius=1,outline='blue', fill='blue')
    
    st.session_state.images.append({'caption':'clustering', 'image':img})

