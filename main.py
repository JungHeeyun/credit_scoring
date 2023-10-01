import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

### 데이터 불러오기 및 전처리 ###
def load_and_preprocess_data():
    df = pd.read_csv("csvdata (3).csv")
    df = df.drop(["first_funding_at", "last_funding_at"], axis=1)
    df = df.drop(columns=df.columns[0])

    industry_dummies_df = pd.get_dummies(df["industry"], drop_first=True)
    country_dummies_df = pd.get_dummies(df["country_code"], drop_first=True)
    region_dummies_df = pd.get_dummies(df["region"], drop_first=True)

    X_col_num = ['homepage_url', "funding_total_usd", 'funding_rounds', 'founded_quarter']
    X_df_num = df[X_col_num]
    X_df = X_df_num.merge(industry_dummies_df, left_index=True, right_index=True).merge(country_dummies_df, left_index=True, right_index=True).merge(region_dummies_df, left_index=True, right_index=True)
    X_df['intercept'] = 1
    y_df = df.label

    return X_df, y_df, df

### 모델 정의 ###
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 6),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x).squeeze()

### 모델 로드 ###
def load_model(X_df, y_df):
    scaler = StandardScaler()
    X_temp_df, _, y_temp_df, _ = train_test_split(X_df, y_df, test_size=0.2, random_state=40, stratify=y_df)
    X_train_df, X_val_df, y_train_df, y_val_df = train_test_split(X_temp_df, y_temp_df, test_size=0.25, random_state=40, stratify=y_temp_df)
    X_train_scaled_df = pd.DataFrame(scaler.fit_transform(X_train_df.values), columns=X_df.columns)

    model_path = "mlp_model.pth"
    loaded_model = MLP(input_size=X_train_scaled_df.shape[1])
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.eval()

    return loaded_model, scaler  # 스케일러 반환

### 예측 함수 ###
def predict(model, input_data):
    model.eval()
    input_tensor = torch.tensor([input_data], dtype=torch.float32)

    with torch.no_grad():
        outputs = model(input_tensor)

    return outputs.numpy().argmax()  # 가장 큰 값의 인덱스를 반환합니다.


### 사용자 입력 얻기 ###
def get_user_input(df):
    # 사용자에게 홈페이지 URL 유무 체크박스 제공
    has_homepage_url = st.checkbox("**Has Homepage URL?**")
    # 체크박스의 선택 여부에 따라 homepage_url 값 설정
    homepage_url = 1 if has_homepage_url else 0
    funding_total_usd = st.number_input("Total Funding in USD", 0)
    funding_rounds = st.number_input("Number of Funding Rounds", 0)
    founded_quarter = st.number_input("Founded Quarter", 1)

    industry_options_display = [industry for industry in df["industry"].unique() if industry != "Unknown" and industry != "0_other_cat"]
    industry_options_display.append("Others")

    selected_industry = st.selectbox("Industry", industry_options_display)

    if selected_industry == "Others":
        selected_industry = "0_other_cat"

    country_code = st.selectbox("Country Code", df["country_code"].unique())
    region = st.selectbox("Region", df["region"].unique())

    user_data = {'homepage_url': homepage_url,
                 'funding_total_usd': funding_total_usd,
                 'funding_rounds': funding_rounds,
                 'founded_quarter': founded_quarter,
                 'industry': selected_industry,
                 'country_code': country_code,
                 'region': region}

    return pd.DataFrame([user_data])

def preprocess_user_input(user_input, df, X_df, scaler):   # scaler 인자 추가
    # Dummy variable creation for user input
    industry_dummies = pd.get_dummies(user_input["industry"], prefix='industry')
    country_dummies = pd.get_dummies(user_input["country_code"], prefix='country_code')
    region_dummies = pd.get_dummies(user_input["region"], prefix='region')

    # Concatenate the user input numerical data with the dummies
    user_input_dummies = pd.concat([user_input[['homepage_url', "funding_total_usd", 'funding_rounds', 'founded_quarter']],
                                    industry_dummies, country_dummies, region_dummies], axis=1)

    # Ensure that all columns from the original data are present in the user input
    # For missing columns, fill with zeros
    for column in X_df.columns:
        if column not in user_input_dummies.columns:
            user_input_dummies[column] = 0

    # Order columns to match the original structure
    user_input_dummies = user_input_dummies[X_df.columns]
    user_input_dummies_scaled = scaler.transform(user_input_dummies)

    return  user_input_dummies_scaled

### 메인 함수 ###
def main():
    st.title("Credit Prediction App")
    X_df, y_df, df = load_and_preprocess_data()

    loaded_model, scaler = load_model(X_df, y_df)  # 스케일러 로드
    user_input = get_user_input(df)
    
    processed_data = preprocess_user_input(user_input, df, X_df, scaler)  # 스케일링 적용

    predicted_class_index = predict(loaded_model, processed_data[0])

    st.subheader("Prediction Result:")
    st.write(f"Predicted Class: {predicted_class_index}")  # 수정된 출력

if __name__ == "__main__":
    main()
