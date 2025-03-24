# 📌 Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Title
st.title("🏨 Hotel Booking Data Analysis")

# ✅ Load the Dataset
file_path = "hotel_bookings 2.csv"
df = pd.read_csv(file_path)

# ✅ Data Preprocessing
df.fillna({"children": 0, "country": "Unknown", "agent": 0}, inplace=True)
df.drop(columns=["company"], inplace=True)
df["reservation_status_date"] = pd.to_datetime(df["reservation_status_date"], dayfirst=True)
df = df[df["adr"] > 0]

st.success("✅ Data Loaded and Preprocessed Successfully!")

# Sidebar - Choose Analysis
analysis_type = st.sidebar.selectbox("Choose Analysis", [
    "Booking Cancellation", "Monthly Revenue", "Revenue by Customer Type",
    "Booking Trends by Market Segment", "ML Model Prediction"
])

if analysis_type == "Booking Cancellation":
    st.header("📊 Booking Cancellation Distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x="is_canceled", data=df, hue="is_canceled", palette=["green", "red"], legend=False)

    ax.set_xlabel("Booking Canceled (0 = No, 1 = Yes)")
    ax.set_ylabel("Count")
    ax.set_title("Booking Cancellation Distribution")
    st.pyplot(fig)

elif analysis_type == "Monthly Revenue":
    st.header("💰 Monthly Average ADR")
    df["arrival_date_month"] = pd.Categorical(df["arrival_date_month"], 
                                              categories=['January', 'February', 'March', 'April', 'May', 'June',
                                                          'July', 'August', 'September', 'October', 'November', 'December'],
                                              ordered=True)
    monthly_revenue = df.groupby("arrival_date_month")["adr"].mean().reindex(df["arrival_date_month"].cat.categories)
    st.bar_chart(monthly_revenue)

elif analysis_type == "Revenue by Customer Type":
    st.header("🛎️ Revenue by Customer Type")
    customer_revenue = df.groupby("customer_type")["adr"].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=customer_revenue.index, y=customer_revenue.values, palette="viridis", ax=ax)
    ax.set_xlabel("Customer Type")
    ax.set_ylabel("Average Daily Rate (ADR)")
    ax.set_title("Revenue by Customer Type")
    st.pyplot(fig)

elif analysis_type == "Booking Trends by Market Segment":
    st.header("📈 Booking Trends by Market Segment")
    market_segment_counts = df["market_segment"].value_counts(normalize=True) * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=market_segment_counts.index, y=market_segment_counts.values, palette="coolwarm", ax=ax)
    ax.set_xlabel("Market Segment")
    ax.set_ylabel("Percentage of Bookings")
    ax.set_title("Booking Trends by Market Segment")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

elif analysis_type == "ML Model Prediction":
    st.header("🤖 Machine Learning Model: Booking Cancellation Prediction")

    # Feature Selection & Encoding
    features = ["lead_time", "arrival_date_week_number", "arrival_date_day_of_month",
                "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children",
                "babies", "meal", "market_segment", "distribution_channel", "reserved_room_type",
                "deposit_type", "customer_type", "adr", "required_car_parking_spaces", "total_of_special_requests"]

    df_encoded = df.copy()
    label_encoders = {}
    for col in ["meal", "market_segment", "distribution_channel", "reserved_room_type", "deposit_type", "customer_type"]:
        label_encoders[col] = LabelEncoder()
        df_encoded[col] = label_encoders[col].fit_transform(df[col])

    X = df_encoded[features]
    y = df_encoded["is_canceled"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train Logistic Regression Model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"✅ Model Accuracy: {accuracy:.2f}")

    # Display Confusion Matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    st.pyplot(fig)

st.success("✅ Analysis Completed!")
