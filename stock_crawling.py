import pandas as pd 
import numpy as np
import plotly.offline as offline 
import plotly.graph_objs as go 

#종목 이름과 종목 코드 불러옴
# header = 0 가져온 table의 첫 행을 DataFrame의 컬럼으로 설정
# [0] 지정한 URL의 첫번째 table을 DataFrame으로 반환
code_df = pd.read_html('http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13', header=0)[0] 

# 종목코드가 6자리이기 때문에 6자리를 맞춰주기 위해 설정해줌 
#code_df["종목코드"] = code_df["종목코드"].map('{:06d}'.format) 
code_df.종목코드 = code_df.종목코드.map('{:06d}'.format) 

# 우리가 필요한 것은 회사명과 종목코드이기 때문에 필요없는 column들은 제외해준다. 
code_df = code_df[['회사명', '종목코드']] 

# 한글로된 컬럼명을 영어로 바꿔준다. 
code_df = code_df.rename(columns={'회사명':'name', '종목코드':'code'}) 
code_df.head()

# 종목 이름을 입력하면 종목에 해당하는 코드를 불러와 # 네이버 금융(http://finance.naver.com)에 넣어줌 
def get_url(item_name, code_df): 
    code = code_df.query("name=='{}'".format(item_name))['code'].to_string(index=False) 
    url = 'http://finance.naver.com/item/sise_day.nhn?code={code}'.format(code=code.strip()) 
    
    print("요청 URL = {}".format(url)) 
    return url 
    
# 일자데이터 url 가져오기 
item_name='삼성전자' 
url = get_url(item_name, code_df) 

# 일자 데이터를 담을 df라는 DataFrame 정의 
df = pd.DataFrame() 

# 1페이지에서 20페이지의 데이터만 가져오기 
for page in range(1, 51): 
    pg_url = '{url}&page={page}'.format(url=url, page=page) 
    df = df.append(pd.read_html(pg_url, header=0)[0], ignore_index=True) 
    
# df.dropna()를 이용해 결측값 있는 행 제거 
df = df.dropna() 

# 상위 5개 데이터 확인하기 
df.head()

# 한글로 된 컬럼명을 영어로 바꿔줌 
df = df.rename(columns= {'날짜': 'date', '종가': 'close', '전일비': 'diff', '시가': 'open', '고가': 'high', '저가': 'low', '거래량': 'volume'}) 

# 데이터의 타입을 int형으로 바꿔줌 
df[['close', 'diff', 'open', 'high', 'low', 'volume']] = df[['close', 'diff', 'open', 'high', 'low', 'volume']].astype(int) 

   
# 컬럼명 'date'의 타입을 date로 바꿔줌 
df['date'] = pd.to_datetime(df['date']) 

# 일자(date)를 기준으로 오름차순 정렬 
df = df.sort_values(by=['date'], ascending=True) 

# 상위 5개 데이터 확인 
df.head()

#plotly
trace = go.Scatter( x=df.date, y=df.close, name=item_name) 
data = [trace] 

layout = dict( 
            title='{}의 종가(close) Time Series'.format(item_name), 
            xaxis=dict( 
                rangeselector=dict( 
                    buttons=list([ 
                        dict(count=1, label='1m', step='month', stepmode='backward'), 
                        dict(count=3, label='3m', step='month', stepmode='backward'), 
                        dict(count=6, label='6m', step='month', stepmode='backward'), 
                        dict(step='all') 
                    ]) 
                ), 
                rangeslider=dict(), 
                type='date' 
            ) 
        ) 
fig = go.Figure(data=data, layout=layout) 
offline.iplot(fig)

# data를 제외한 데이터를 numpy형식으로 저장
df1 = pd.DataFrame() 
df1 = df[['close', 'diff', 'open', 'high', 'low', 'volume']] 
df1 = df1.to_numpy(dtype="Int64")
np.save('C:/Users/JPark/Documents/Python Scripts/Stock/samsung.npy', arr=df1)
#df2.to_csv('C:/Users/JPark/Documents/Python Scripts/Stock/date.csv')

# data를 csv 형식으로 저장
df2 = df['date']
df2.to_csv('C:/Users/JPark/Documents/Python Scripts/Stock/date.csv',
                 sep=',',
                 na_rep='NaN', 
                 float_format = '%.2f', # 2 decimal places
                 columns = ['date'], # columns to write
                 index = False) # do not write index



#출처: https://excelsior-cjh.tistory.com/109?category=975542 [EXCELSIOR]
