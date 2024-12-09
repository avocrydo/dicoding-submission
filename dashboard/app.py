import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Define color scheme
COLORS = {
    "primary": "#2B6FB3",
    "secondary": "#E24A33",
    "tertiary": "#7AB51D",
    "light": "#A8D5FF",
    "highlight": "#FFB31A",
    "casual": "#FF9999",
    "registered": "#66B3FF",
}


@st.cache_data
def load_data():
    data_path = Path(__file__).parents[1] / "data" / "day.csv"
    data = pd.read_csv(data_path)
    data["dteday"] = pd.to_datetime(data["dteday"])
    return data


def main():
    data = load_data()

    # Sidebar
    st.sidebar.header("Navigation")
    analysis_type = st.sidebar.radio(
        "Choose Analysis:", ["Weather Impact", "Usage Patterns", "Future Demand"]
    )

    if analysis_type == "Weather Impact":
        st.header("Weather Impact Analysis")

        # Weather condition impact
        st.subheader("Rental Distribution by Weather Condition")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(
            x="weathersit",
            y="cnt",
            data=data,
            palette=[COLORS["light"], COLORS["primary"], COLORS["secondary"]],
        )
        weather_labels = [
            f"Clear\n(Avg: {data[data['weathersit']==1]['cnt'].mean():.0f})",
            f"Mist/Cloudy\n(Avg: {data[data['weathersit']==2]['cnt'].mean():.0f})",
            f"Light Snow/Rain\n(Avg: {data[data['weathersit']==3]['cnt'].mean():.0f})",
        ]
        plt.xticks(range(3), weather_labels)
        st.pyplot(fig)

        # Temperature impact
        st.subheader("Temperature Impact on Rentals")
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.regplot(
                data=data,
                x="temp",
                y="cnt",
                scatter_kws={"alpha": 0.5, "color": COLORS["primary"]},
                line_kws={"color": COLORS["secondary"]},
            )
            correlation = data["temp"].corr(data["cnt"])
            plt.text(
                0.05,
                plt.ylim()[1] * 0.9,
                f"Correlation: {correlation:.2f}",
                bbox=dict(facecolor="white", alpha=0.8),
            )
            st.pyplot(fig)

        with col2:
            temp_ranges = pd.qcut(
                data["temp"], q=4, labels=["Cold", "Cool", "Warm", "Hot"]
            )
            temp_stats = data.groupby(temp_ranges)["cnt"].agg(["mean", "std"])
            fig, ax = plt.subplots(figsize=(8, 6))
            plt.bar(
                temp_stats.index,
                temp_stats["mean"],
                yerr=temp_stats["std"],
                capsize=5,
                color=COLORS["secondary"],
            )
            st.pyplot(fig)

    elif analysis_type == "Usage Patterns":
        st.header("Usage Pattern Analysis")

        # Weekly patterns
        st.subheader("Weekly Usage Patterns")

        # First visualization: Overall weekly pattern
        fig, ax = plt.subplots(figsize=(12, 6))
        weekly_pattern = data.groupby("weekday")["cnt"].agg(["mean", "std"]).round(0)
        days = [
            "Sunday",
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
        ]

        plt.errorbar(
            range(len(days)),
            weekly_pattern["mean"],
            yerr=weekly_pattern["std"],
            fmt="o-",
            capsize=5,
            color=COLORS["primary"],
            linewidth=2,
            markersize=8,
        )

        plt.xticks(range(len(days)), days, rotation=45)
        plt.title("Weekly Rental Pattern with Variation")
        plt.xlabel("Day of Week")
        plt.ylabel("Average Daily Rentals")
        plt.grid(True, alpha=0.3)

        # Add value labels
        for i, v in enumerate(weekly_pattern["mean"]):
            plt.text(
                i, v + weekly_pattern["std"][i], f"{int(v)}", ha="center", va="bottom"
            )

        st.pyplot(fig)

        # User type distribution across week
        st.subheader("User Types Distribution Across Week")
        weekly_user_types = data.groupby("weekday")[["casual", "registered"]].mean()

        fig, ax = plt.subplots(figsize=(12, 6))
        # Clear the previous plot
        plt.clf()
        # Create new axis
        ax = fig.add_subplot(111)

        # Plot bars for each user type
        x = np.arange(len(days))
        width = 0.35
        ax.bar(
            x - width / 2,
            weekly_user_types["casual"],
            width,
            label="Casual",
            color=COLORS["casual"],
        )
        ax.bar(
            x + width / 2,
            weekly_user_types["registered"],
            width,
            label="Registered",
            color=COLORS["registered"],
        )

        # Customize the plot
        ax.set_title("User Types Distribution Across Week")
        ax.set_xlabel("Day of Week")
        ax.set_ylabel("Average Number of Rentals")
        ax.set_xticks(x)
        ax.set_xticklabels(days, rotation=45)
        ax.legend(title="User Type")
        ax.grid(True, alpha=0.3)

        # Add value labels
        for i, container in enumerate(
            [weekly_user_types["casual"], weekly_user_types["registered"]]
        ):
            for j, value in enumerate(container):
                ax.text(
                    j + (width / 2 if i else -width / 2),
                    value,
                    f"{int(value)}",
                    ha="center",
                    va="bottom",
                )

        plt.tight_layout()
        st.pyplot(fig)

        # Working vs Non-working days analysis
        st.subheader("Working vs Non-working Day Patterns")

        # Calculate statistics
        workday_stats = data.groupby("workingday")[["casual", "registered"]].mean()

        fig, ax = plt.subplots(figsize=(10, 6))
        # Clear the previous plot
        plt.clf()
        # Create new axis
        ax = fig.add_subplot(111)

        # Plot bars for each user type
        x = np.arange(2)  # 0 for non-working, 1 for working
        width = 0.35
        ax.bar(
            x - width / 2,
            workday_stats["casual"],
            width,
            label="Casual",
            color=COLORS["casual"],
        )
        ax.bar(
            x + width / 2,
            workday_stats["registered"],
            width,
            label="Registered",
            color=COLORS["registered"],
        )

        # Customize the plot
        ax.set_title("Average Rentals: Working vs Non-working Days")
        ax.set_xlabel("Day Type")
        ax.set_ylabel("Average Number of Rentals")
        ax.set_xticks(x)
        ax.set_xticklabels(["Non-working", "Working"])
        ax.legend(title="User Type")
        ax.grid(True, alpha=0.3)

        # Add value labels
        for i, container in enumerate(
            [workday_stats["casual"], workday_stats["registered"]]
        ):
            for j, value in enumerate(container):
                ax.text(
                    j + (width / 2 if i else -width / 2),
                    value,
                    f"{int(value)}",
                    ha="center",
                    va="bottom",
                )

        plt.tight_layout()
        st.pyplot(fig)

        # Add summary metrics
        st.subheader("Key Usage Statistics")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Weekday Average",
                f"{data[data['workingday'] == 1]['cnt'].mean():.0f}",
                f"{data[data['workingday'] == 1]['cnt'].std():.0f} std",
            )
        with col2:
            st.metric(
                "Weekend Average",
                f"{data[data['workingday'] == 0]['cnt'].mean():.0f}",
                f"{data[data['workingday'] == 0]['cnt'].std():.0f} std",
            )
        with col3:
            peak_day = days[data.groupby("weekday")["cnt"].mean().idxmax()]
            st.metric("Peak Usage Day", peak_day)

    else:  # Future Demand
        st.header("Future Demand Analysis")

        # Seasonal patterns
        st.subheader("Seasonal Patterns")
        season_data = data.groupby("season")[["casual", "registered"]].mean()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        season_data.plot(
            kind="bar", ax=ax1, color=[COLORS["casual"], COLORS["registered"]]
        )
        ax1.set_title("Average Daily Rentals by Season")

        seasonal_pct = data.groupby("season")["cnt"].sum() / data["cnt"].sum() * 100
        seasonal_pct.plot(
            kind="pie",
            autopct="%1.1f%%",
            ax=ax2,
            colors=[
                COLORS["light"],
                COLORS["primary"],
                COLORS["secondary"],
                COLORS["tertiary"],
            ],
        )
        st.pyplot(fig)

        # Growth analysis
        st.subheader("Long-term Growth Trends")
        yearly_rentals = data.groupby(data["dteday"].dt.year)["cnt"].sum()
        yearly_growth = yearly_rentals.pct_change() * 100

        fig, ax = plt.subplots(figsize=(12, 6))
        plt.semilogy(
            yearly_rentals.index, yearly_rentals.values, "o-", color=COLORS["primary"]
        )

        for i in range(len(yearly_rentals)):
            value = yearly_rentals.values[i]
            plt.text(
                yearly_rentals.index[i],
                value,
                f"{value:,.0f}",
                ha="center",
                va="bottom",
            )
        st.pyplot(fig)

        # Add metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rentals", f"{data['cnt'].sum():,}")
        with col2:
            st.metric("Average Daily Rentals", f"{data['cnt'].mean():.0f}")
        with col3:
            st.metric("Growth Rate", f"{yearly_growth.mean():.1f}%")


if __name__ == "__main__":
    main()
