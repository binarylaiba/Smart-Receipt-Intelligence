"""Streamlit Interactive Dashboard for AI Receipt Analyser.

Integrates Computer Vision Preprocessing (OpenCV), OCR (EasyOCR),
Financial Contract Structuring (Groq LLM + Pydantic v2), and
Analytics (Pandas & Plotly).
"""

import os
import io
import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import project modules
from ocr_engine import preprocess_image, extract_raw_text
from llm_parser import parse_receipt_with_groq, ReceiptAnalysis
from analytics import compute_metrics, verify_arithmetic


# --- Page Configuration & Styling ---
st.set_page_config(
    page_title="AI Receipt Analyser & Financial Assistant",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished aesthetic
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    # --- Header Section ---
    st.markdown('<div class="main-header">🧾 AI Receipt Analyser & Financial Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Upload any receipt to automatically extract items, verify arithmetic, categorize expenditures, and generate intelligent budget recommendations.</div>',
        unsafe_allow_html=True,
    )

    # --- Sidebar Configuration ---
    with st.sidebar:
        st.header("⚙️ System Status")
        st.success("🟢 AI Pipeline Ready")

        st.markdown("---")
        st.markdown("### 🛠️ Architecture Stack")
        st.markdown(
            """
            - **Vision Preprocessing**: OpenCV Bilateral & Adaptive Gaussian Threshold
            - **OCR Engine**: EasyOCR (Local)
            - **LLM Reasoning**: Groq Llama / GPT-OSS
            - **Data Contract**: Pydantic v2
            - **Analytics**: Pandas & Plotly Express
            """
        )
        st.markdown("---")
        st.markdown("### 💡 Quick Tips")
        st.markdown(
            """
            1. Upload a clear, flat receipt image.
            2. Avoid severe glare and crumpled corners.
            3. Check the **Itemized List** tab for arithmetic validation.
            """
        )

    # --- File Upload Section ---
    uploaded_file = st.file_uploader(
        "Upload Receipt Image (PNG, JPG, JPEG)",
        type=["png", "jpg", "jpeg"],
        help="Upload a clear photo or scan of your receipt.",
    )

    if uploaded_file is not None:
        try:
            image_bytes = uploaded_file.getvalue()
            pil_image = Image.open(io.BytesIO(image_bytes))

            # --- Layout: Two Main Columns ---
            col_left, col_right = st.columns([1, 1.2], gap="large")

            # --- Left Column: Image & OCR Pipeline ---
            with col_left:
                st.subheader("📷 Image Processing Pipeline")
                st.image(pil_image, caption="Original Uploaded Receipt", use_container_width=True)

                # Process Image & OCR with caching / progress feedback
                with st.spinner("⚡ Running Computer Vision & EasyOCR..."):
                    preprocessed_np, preprocessed_pil = preprocess_image(image_bytes)
                    raw_ocr_text = extract_raw_text(preprocessed_np)

                # Expander for Preprocessed Binary Image
                with st.expander("🔍 View Preprocessed Binary Image (OpenCV Filtered)", expanded=False):
                    st.image(
                        preprocessed_pil,
                        caption="Grayscale + Bilateral Filter + Adaptive Gaussian Threshold",
                        use_container_width=True,
                    )

                # Expander for Raw OCR Text
                with st.expander("📝 View Raw OCR Extracted Text", expanded=False):
                    if raw_ocr_text.strip():
                        st.text_area("OCR Text Stream", raw_ocr_text, height=220)
                    else:
                        st.info("No text detected by OCR engine.")

            # --- Right Column: Analytics & LLM Insights ---
            with col_right:
                st.subheader("📈 Financial Insights & Audit")

                if not raw_ocr_text.strip():
                    st.error("❌ Could not extract readable text from the image. Please try a clearer receipt.")
                    return

                # Run LLM Structuring (reads key securely from .env in the backend)
                with st.spinner("🤖 Structuring with AI & Validating Contract..."):
                    try:
                        receipt_data: ReceiptAnalysis = parse_receipt_with_groq(
                            raw_ocr_text=raw_ocr_text,
                        )
                    except Exception as llm_err:
                        st.error(f"Failed to parse receipt with LLM: {llm_err}")
                        return

                # --- 1. Top KPI Metrics ---
                kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
                with kpi_col1:
                    st.metric(
                        label="🏪 Merchant",
                        value=receipt_data.merchant or "Unknown",
                        delta=f"📅 {receipt_data.date}" if receipt_data.date else None,
                        delta_color="off",
                    )
                with kpi_col2:
                    st.metric(
                        label="💵 Total Paid",
                        value=f"${receipt_data.total:.2f}",
                    )
                with kpi_col3:
                    st.metric(
                        label="🧾 Subtotal / Tax",
                        value=f"${receipt_data.subtotal:.2f}",
                        delta=f"Tax: ${receipt_data.tax:.2f}",
                        delta_color="off",
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                # --- Compute Analytics with Pandas ---
                items_df, category_summary_df = compute_metrics(receipt_data)
                audit_report = verify_arithmetic(receipt_data)

                # --- 2. Interactive Tabs ---
                tab1, tab2, tab3 = st.tabs([
                    "📊 Spending Breakdown",
                    "🧾 Itemized List",
                    "💡 AI Financial Advice",
                ])

                # --- Tab 1: Spending Breakdown & Donut Chart ---
                with tab1:
                    if not category_summary_df.empty:
                        # Color palette tailored for financial categories
                        category_colors = {
                            "Produce": "#10B981",
                            "Dairy": "#38BDF8",
                            "Meat": "#EF4444",
                            "Bakery": "#F59E0B",
                            "Snacks": "#EC4899",
                            "Beverages": "#6366F1",
                            "Household": "#8B5CF6",
                            "Personal Care": "#14B8A6",
                            "Other": "#94A3B8",
                        }

                        fig = px.pie(
                            category_summary_df,
                            names="category",
                            values="total_spent",
                            hole=0.42,
                            title="Expenditure Distribution by Category",
                            color="category",
                            color_discrete_map=category_colors,
                        )
                        fig.update_traces(
                            textposition="inside",
                            textinfo="percent+label",
                            hovertemplate="<b>%{label}</b><br>Spent: $%{value:.2f}<br>Share: %{percent}",
                        )
                        fig.update_layout(
                            showlegend=True,
                            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
                            margin=dict(t=40, b=20, l=10, r=10),
                            height=360,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Category Summary Table
                        st.markdown("##### Category Breakdown Table")
                        st.dataframe(
                            category_summary_df.style.format({
                                "total_spent": "${:.2f}",
                                "percentage": "{:.1f}%",
                                "item_count": "{:.0f}",
                            }),
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.info("No category data available to plot.")

                # --- Tab 2: Itemized List ---
                with tab2:
                    if not items_df.empty:
                        st.markdown(f"**Total Items Identified:** `{len(items_df)}`")
                        st.dataframe(
                            items_df.style.format({
                                "price": "${:.2f}",
                                "quantity": "{:.1f}",
                            }),
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.info("No line items extracted.")

                    # Audit Check section
                    with st.expander("🔍 Arithmetic Verification Audit", expanded=False):
                        if audit_report["is_subtotal_accurate"] and audit_report["is_total_accurate"]:
                            st.success("✅ Arithmetic Verified: Itemized sum matches declared subtotal and total.")
                        else:
                            st.warning("⚠️ Potential Math Discrepancy on Receipt:")
                            for flag in audit_report["discrepancy_flags"]:
                                st.write(f"- {flag}")

                # --- Tab 3: AI Financial Advice & Intelligence ---
                with tab3:
                    # Spending Summary
                    st.markdown("##### 📌 Spending Summary")
                    st.info(receipt_data.spending_summary or "No summary available.")

                    # Discretionary Flags
                    st.markdown("##### ⚠️ Discretionary & Non-Essential Items")
                    if receipt_data.discretionary_flags:
                        for flag in receipt_data.discretionary_flags:
                            st.warning(f"• {flag}")
                    else:
                        st.success("🎉 No high-cost discretionary or impulse items detected.")

                    # Budget Advice
                    st.markdown("##### 💡 Personalized Money-Saving Tips")
                    if receipt_data.budget_advice:
                        for idx, tip in enumerate(receipt_data.budget_advice, 1):
                            st.markdown(f"**{idx}.** {tip}")
                    else:
                        st.write("No specific budget recommendations generated.")

        except Exception as e:
            st.error(f"An unexpected error occurred during analysis: {e}")

    else:
        # Prompt state when no file is uploaded
        st.info("👆 Please upload a receipt image from the file uploader above to begin analysis.")


if __name__ == "__main__":
    main()
