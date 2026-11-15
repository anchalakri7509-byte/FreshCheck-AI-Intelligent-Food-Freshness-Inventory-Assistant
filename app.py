                tf.__version__
            )

        except Exception as error:

            st.error(
                f"Could not read TensorFlow version: {error}"
            )

        st.divider()

        st.write("**Expected CNN class order:**")

        for index, class_name in enumerate(CLASSES):

            st.write(
                f"{index} = {class_name}"
            )

        st.divider()

        st.write("**CNN preprocessing layers:**")

        rescaling_found = False

        for layer in cnn.layers:

            if layer.__class__.__name__ == "Rescaling":

                rescaling_found = True

                st.write({
                    "layer_name": layer.name,
                    "layer_type": layer.__class__.__name__,
                    "configuration": layer.get_config()
                })

        if rescaling_found:

            st.success(
                "Rescaling layer detected. The app passes "
                "pixel values in the 0–255 range."
            )

        else:

            st.warning(
                "No top-level Rescaling layer detected. "
                "Do not change preprocessing until the model "
                "configuration has been checked."
            )

        st.divider()

        st.write("**CNN SHA256 fingerprint:**")

        st.code(cnn_fingerprint)

    st.divider()

    st.subheader("Risk Model")

    if risk_model is None:

        st.error(risk_error)

    else:

        st.success("Risk model loaded successfully.")

        st.write("**Model file:**", str(RISK_PATH))

        st.write(
            "**Input feature order:** Fruit code, days since purchase, "
            "temperature, humidity"
        )

        if hasattr(risk_model, "classes_"):

            st.write(
                "**Risk model classes:**",
                str(list(risk_model.classes_))
            )

        st.write("**Risk model SHA256 fingerprint:**")

        st.code(risk_fingerprint)

    st.divider()

    st.subheader("Deployment File Structure")

    st.code(
        """
FreshCheck_AI/
│
├── app.py
├── requirements.txt
├── README.md
│
└── models/
    ├── freshness_cnn.keras
    └── spoilage_risk.joblib
        """,
        language="text"
    )

    st.caption(
        "The deployed CNN class order must match the class order "
        "used during model training."
    )


# ============================================================
# 18. FOOTER
# ============================================================

st.divider()

st.caption(
    "FreshCheck AI | CNN + Machine Learning + Generative AI + SQLite"
)
