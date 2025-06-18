# --- population_eda.py ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class PopulationEDA:
    def __init__(self):
        st.title("📊 Population Trends Analysis")

        uploaded = st.file_uploader("Upload population_trends.csv", type="csv")
        if uploaded is None:
            st.info("Please upload population_trends.csv")
            return

        df = pd.read_csv(uploaded)

        # 전처리
        df.replace('-', 0, inplace=True)
        df[['인구', '출생아수(명)', '사망자수(명)']] = df[['인구', '출생아수(명)', '사망자수(명)']].astype(int)

        tabs = st.tabs(["Summary", "Trend", "Regional", "Changes", "Visualization"])

        with tabs[0]:
            st.subheader("🔍 Data Overview")
            st.text("DataFrame Info")
            st.text(df.info())
            st.subheader("Descriptive Statistics")
            st.dataframe(df.describe())

        with tabs[1]:
            st.subheader("📈 Yearly Population Trend")
            df_nat = df[df['지역'] == '전국']
            fig, ax = plt.subplots()
            sns.lineplot(data=df_nat, x='연도', y='인구', ax=ax)
            ax.set_title("National Population Trend")
            st.pyplot(fig)

            recent = df_nat.sort_values('연도').tail(3)
            avg_birth = recent['출생아수(명)'].mean()
            avg_death = recent['사망자수(명)'].mean()
            latest = recent.iloc[-1]
            pred_pop = latest['인구'] + (avg_birth - avg_death) * (2035 - latest['연도'])
            ax.axhline(pred_pop, color='gray', linestyle='--')
            ax.text(2035, pred_pop, f"2035 Prediction: {int(pred_pop):,}", color='black')
            st.pyplot(fig)

        with tabs[2]:
            st.subheader("📊 5-Year Population Change by Region")
            years = sorted(df['연도'].unique())
            if len(years) < 6:
                st.warning("Dataset must include at least 6 years of data")
                return

            df_5 = df[df['연도'].isin([years[-1], years[-6]])]
            pivot = df_5.pivot(index='지역', columns='연도', values='인구')
            pivot = pivot.drop('전국', errors='ignore')
            pivot['증가량'] = pivot[years[-1]] - pivot[years[-6]]
            pivot['변화율'] = (pivot['증가량'] / pivot[years[-6]]) * 100

            top_diff = pivot.sort_values('증가량', ascending=False)

            fig1, ax1 = plt.subplots()
            sns.barplot(x='증가량', y=top_diff.index, data=top_diff, ax=ax1)
            ax1.set_title("Population Change (5 years)")
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots()
            sns.barplot(x='변화율', y=top_diff.index, data=top_diff, ax=ax2)
            ax2.set_title("Change Rate (%)")
            st.pyplot(fig2)

        with tabs[3]:
            st.subheader("📈 Top 100 Change Cases")
            df_sorted = df[df['지역'] != '전국'].copy()
            df_sorted['증감'] = df_sorted.groupby('지역')['인구'].diff()
            top100 = df_sorted.nlargest(100, '증감')

            def color_scale(val):
                color = 'background-color: '
                if val > 0:
                    color += '#add8e6'  # 파랑
                elif val < 0:
                    color += '#f4cccc'  # 빨강
                else:
                    color += 'white'
                return color

            styled_df = top100.style.format({'인구': "{:,}", '증감': "{:,}"}).applymap(color_scale, subset=['증감'])
            st.dataframe(styled_df)

        with tabs[4]:
            st.subheader("📊 Heatmap")
            pivot = df.pivot(index='지역', columns='연도', values='인구')
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(pivot, cmap='YlGnBu', annot=False, ax=ax)
            st.pyplot(fig)

            st.markdown(
                "- This heatmap shows yearly population distribution by region.\n"
                "- Brighter colors indicate higher population.\n"
                "- Useful for identifying regional trends over time.")
