import multipage_streamlit as mt
from pages import page_a, page_b
import streamlit as st
from streamlit.elements.image import image_to_url
#from streamlit.server.server import Server
import datetime

st.set_page_config(
    page_title="热门城市房价采集及分析", page_icon=":rainbow:",
    layout='wide', initial_sidebar_state="auto",
)

#加载背景图（本地图片先转url，网页图片就直接给图片的链接）
img_url = 'https://img-blog.csdnimg.cn/d1dd8bb83096492c9891b64267dba6e7.png'
# 通过markdown加载背景图（可以是动图、静图）
st.markdown('''
<style>
.css-fg4pbf {background-image: url(''' + img_url + ''');}</style>
''', unsafe_allow_html=True)


app = mt.MultiPage()
app.add("房地产数据", page_a.run)
app.add("环境数据", page_b.run)
app.add("消费数据", page_b.run)
app.run_radio('经济大数据')

# sessions = Server.get_current()._session_info_by_id
# st.sidebar.info(f'当前在线人数：{len(sessions)}')
# alternatives: app.run_expander() or app.run_radio() if you prefer those
