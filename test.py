import streamlit as st
import pandas as pd
from vnstock3 import Vnstock
from time import sleep

# Thiết lập dữ liệu ban đầu
vn30_symbols = ['VCB', 'ACB', 'TCB', 'BID', 'CTG', 'FPT', 'GAS', 'HPG', 'MBB', 'MSN',
                'MWG', 'NVL', 'PDR', 'PLX', 'POW', 'SAB', 'SSI', 'STB', 'VHM', 'VIC', 'VNM', 'VPB', 'VRE']

# Thiết lập API Vnstock
stock = Vnstock().stock(symbol='VN30F1M', source='VCI')

# Thiết lập giao diện Streamlit
st.title("Bảng Giá Chứng Khoán VN30 Realtime")

# Thiết lập chế độ Auto Update
auto_update = st.checkbox('Auto Update', value=True)

# Thiết lập thời gian cập nhật
update_interval = st.slider('Update Interval (seconds)', min_value=1, max_value=60, value=5)

# Hàm lấy dữ liệu và hiển thị bảng
def show_price_board():
    try:
        # Lấy dữ liệu bảng giá
        price_board_data = stock.trading.price_board(symbols_list=vn30_symbols)
        
        # Chuyển đổi dữ liệu thành DataFrame
        df = pd.DataFrame(price_board_data)

        # Hiển thị cấu trúc dữ liệu để kiểm tra
        st.write("Data columns:", df.columns)
        
        # Kiểm tra và xử lý từng cột một cách linh hoạt
        expected_columns = ['ceiling', 'floor', 'last', 'best_bid', 'best_ask']
        for col in expected_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                st.warning(f"Cột '{col}' không tồn tại trong dữ liệu.")
        
        # Hiển thị dữ liệu lên Streamlit
        st.dataframe(df)
    
    except Exception as e:
        st.error(f"Không thể lấy dữ liệu: {e}")

# Chạy cập nhật bảng giá liên tục nếu Auto Update được bật
if auto_update:
    while True:
        show_price_board()
        sleep(update_interval)
        st.experimental_rerun()
else:
    show_price_board()
