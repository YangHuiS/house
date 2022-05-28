import multipage_streamlit as mt
from pages import page_a, page_b
import streamlit as st
from streamlit.elements.image import image_to_url
from streamlit.server.server import Server
import datetime

st.set_page_config(
    page_title="热门城市房价采集及分析", page_icon=":rainbow:",
    layout='wide', initial_sidebar_state="auto",
)

#加载背景图（本地图片先转url，网页图片就直接给图片的链接）
img_url = image_to_url('back1.png', width=-3, clamp=False,
                       channels='RGB', output_format='auto', image_id='',
                       allow_emoji=False)
# 通过markdown加载背景图（可以是动图、静图）
st.markdown('''
<style>
.css-fg4pbf {background-image: url(''' + img_url + ''');}</style>
''', unsafe_allow_html=True)

# %% ----------会话状态----------
if 'first_visit' not in st.session_state:
    st.session_state.first_visit = True
else:
    st.session_state.first_visit = False
# # 初始化全局配置
# if st.session_state.first_visit:
#     st.session_state.date_time = datetime.datetime.now() + datetime.timedelta(hours=8)  # Streamlit Cloud的时区是UTC，加8小时即北京时间
#     # with open('./back.mp4', 'rb') as f:
#     #     au = f.read()
#     # t1 = datetime.datetime.now()
#     # st.video(au)
#     # st.write('''<style>
#     # #video {controls="controls" autoplay="autoplay" loop}</style>''', unsafe_allow_html=True)
#     # if datetime.datetime.now() - t1 == 20:
#     #     st.stop()
#     st.snow()
#     st.balloons()


app = mt.MultiPage()
app.add("房地产数据", page_a.run)
app.add("环境数据", page_b.run)
app.add("消费数据", page_b.run)
app.run_radio('经济大数据')

sessions = Server.get_current()._session_info_by_id
st.sidebar.info(f'当前在线人数：{len(sessions)}')
# alternatives: app.run_expander() or app.run_radio() if you prefer those