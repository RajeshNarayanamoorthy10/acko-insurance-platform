"""
Acko Insurance Platform - Streamlit app.

Tabs:
  - Chat with Acko   (Module 1: RAG policy chatbot)
  - Get a Quote       (Module 2: premium predictor)
  - Dashboard          (Module 4: added once real usage data exists)
"""

import streamlit as st

from src.rag.chatbot import load_vector_store, answer_question
from src.ml.predict import predict_quote

from datetime import date, timedelta

import pandas as pd
import plotly.express as px
from sqlalchemy import text

from src.db.database import engine

st.set_page_config(page_title="Acko Insurance Platform", layout="wide")
st.title("Acko Insurance - AI Platform")

@st.cache_data(ttl=60)
def load_quotations(start_date, end_date):
    query = text("SELECT * FROM quotations WHERE date(created_at) BETWEEN :start AND :end")
    return pd.read_sql(query, engine, params={"start": str(start_date), "end": str(end_date)})


@st.cache_data(ttl=60)
def load_chat_logs(start_date, end_date):
    query = text("SELECT * FROM chat_logs WHERE date(created_at) BETWEEN :start AND :end")
    return pd.read_sql(query, engine, params={"start": str(start_date), "end": str(end_date)})

tab_chat, tab_quote, tab_dashboard = st.tabs(["Chat with Acko", "Get a Quote", "Dashboard"])

# ----------------------------------------------------------------------
# TAB 1: Chat with Acko (Module 1)
# ----------------------------------------------------------------------
with tab_chat:
    st.subheader("Ask about your policy")

    # Load the vector store once and cache it across reruns - Streamlit
    # reruns the whole script on every interaction, so without caching
    # this would reload the embedding model on every single click.
    @st.cache_resource
    def get_vector_store():
        return load_vector_store()

    store = get_vector_store()

    question = st.text_input("Your question:", key="chat_question")
    if st.button("Ask", key="ask_button") and question:
        with st.spinner("Thinking..."):
            try:
                result = answer_question(store, question)
            except RuntimeError as e:
                st.error(str(e))
                result = None
            except Exception as e:
                st.error(f"Something went wrong while getting an answer: {e}")
                result = None

        if result:
            st.write(result["answer"])
            with st.expander("Sources"):
                for s in result["sources"]:
                    st.caption(f"{s['source']}, page {s['page']}")

# ----------------------------------------------------------------------
# TAB 2: Get a Quote (Module 2)
# ----------------------------------------------------------------------
with tab_quote:
    st.subheader("Get your premium quote")

    col1, col2, col3 = st.columns(3)
    with col1:
        vehicle_type = st.selectbox("Vehicle type", ["car", "bike"])
        vehicle_make = st.text_input("Make (e.g. Hyundai, Honda)", "Hyundai")
        segment = st.text_input("Segment (e.g. Hatchback, Sedan)", "Hatchback")
        fuel_type = st.selectbox("Fuel type", ["Petrol", "Diesel", "CNG", "Electric"])
        policy_type = st.selectbox("Policy type", ["Comprehensive", "Third Party"])

    with col2:
        customer_age = st.number_input("Your age", 18, 80, 35)
        city_tier = st.selectbox("City tier", [1, 2, 3])
        city_risk_score = st.slider("City risk score", 0.0, 1.0, 0.5)
        manufacturing_year = st.number_input("Manufacturing year", 2000, 2026, 2021)
        vehicle_age_years = st.number_input("Vehicle age (years)", 0, 25, 4)

    with col3:
        engine_cc = st.number_input("Engine CC", 50, 5000, 1197)
        idv = st.number_input("IDV (Rs.)", 10000, 5000000, 450000)
        ncb_percent = st.slider("NCB %", 0, 50, 20)
        claim_history_count = st.number_input("Past claims", 0, 10, 0)
        num_addons = st.number_input("Number of add-ons", 0, 10, 2)

    if st.button("Get Quote"):
        inputs = {
            "customer_age": customer_age,
            "city_tier": city_tier,
            "city_risk_score": city_risk_score,
            "manufacturing_year": manufacturing_year,
            "vehicle_age_years": vehicle_age_years,
            "engine_cc": engine_cc,
            "idv": idv,
            "ncb_percent": ncb_percent,
            "claim_history_count": claim_history_count,
            "num_addons": num_addons,
            "vehicle_type": vehicle_type,
            "vehicle_make": vehicle_make,
            "segment": segment,
            "fuel_type": fuel_type,
            "policy_type": policy_type,
        }
        premium = predict_quote(inputs)
        st.success(f"Estimated annual premium: Rs.{premium:,.2f}")

# ----------------------------------------------------------------------
# TAB 3: Dashboard (Module 4)
# ----------------------------------------------------------------------
with tab_dashboard:
    st.subheader("Management Dashboard")

    col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
    with col_f1:
        start_date = st.date_input("From", value=date.today() - timedelta(days=30))
    with col_f2:
        end_date = st.date_input("To", value=date.today())
    with col_f3:
        st.write("")
        st.write("")
        if st.button("Refresh"):
            st.cache_data.clear()
            st.rerun()

    quotations_df = load_quotations(start_date, end_date)
    chats_df = load_chat_logs(start_date, end_date)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Quotations", len(quotations_df))
    k2.metric(
        "Avg Premium Quoted",
        f"Rs.{quotations_df['predicted_premium'].mean():,.0f}" if len(quotations_df) else "Rs.0",
    )
    k3.metric("Total Chat Questions", len(chats_df))
    k4.metric(
        "Avg Response Time",
        f"{chats_df['response_time_seconds'].mean():.1f}s" if len(chats_df) else "0s",
    )

    st.divider()
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        if len(quotations_df) > 0:
            daily = quotations_df.copy()
            daily["date"] = pd.to_datetime(daily["created_at"]).dt.date
            daily_counts = daily.groupby("date").size().reset_index(name="count")
            fig = px.line(daily_counts, x="date", y="count", markers=True,
                          title="Quotations Generated Over Time")
            fig.update_traces(line_color="#3B82F6")
            fig.update_layout(yaxis_title="Number of Quotes", xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No quotations in this date range.")

    with chart_col2:
        if len(quotations_df) > 0:
            by_type = quotations_df.groupby("vehicle_type")["predicted_premium"].mean().reset_index()
            fig = px.bar(by_type, x="vehicle_type", y="predicted_premium",
                        title="Avg Premium by Vehicle Type", color="vehicle_type",
                        color_discrete_sequence=["#3B82F6", "#F59E0B"])
            fig.update_layout(yaxis_title="Avg Premium (Rs.)", xaxis_title="", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No quotations in this date range.")

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        if len(quotations_df) > 0:
            by_policy = quotations_df.groupby("policy_type")["predicted_premium"].mean().reset_index()
            fig = px.bar(by_policy, x="policy_type", y="predicted_premium",
                        title="Avg Premium by Policy Type", color="policy_type",
                        color_discrete_sequence=["#3B82F6", "#F59E0B"])
            fig.update_layout(yaxis_title="Avg Premium (Rs.)", xaxis_title="", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No quotations in this date range.")

    with chart_col4:
        if len(chats_df) > 0:
            daily_chats = chats_df.copy()
            daily_chats["date"] = pd.to_datetime(daily_chats["created_at"]).dt.date
            daily_chat_counts = daily_chats.groupby("date").size().reset_index(name="count")
            fig = px.line(daily_chat_counts, x="date", y="count", markers=True,
                          title="Chat Questions Over Time")
            fig.update_traces(line_color="#F59E0B")
            fig.update_layout(yaxis_title="Number of Questions", xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No chat questions in this date range.")

    st.divider()
    st.subheader("Recent Chat Questions")
    if len(chats_df) > 0:
        recent = chats_df.sort_values("created_at", ascending=False)[
            ["created_at", "question", "response_time_seconds"]
        ].head(10)
        st.dataframe(recent, use_container_width=True, hide_index=True)
    else:
        st.info("No chat questions in this date range.")