def get_lesion_location(bbox, image_shape=(224, 224)):
    """Returns a plain-language description of where the flagged area is."""

    if bbox is None:
        return "No suspicious area found"

    x, y, w, h = bbox

    img_w = image_shape[1]
    img_h = image_shape[0]

    cx = x + w / 2
    cy = y + h / 2

    # Horizontal position
    if cx < img_w / 3:
        horizontal = "left"
    elif cx < 2 * img_w / 3:
        horizontal = "center"
    else:
        horizontal = "right"

    # Vertical position
    if cy < img_h / 3:
        vertical = "upper"
    elif cy < 2 * img_h / 3:
        vertical = "middle"
    else:
        vertical = "lower"

    # Plain-language phrasing instead of "Middle Center Oral Region"
    location_phrases = {
        ("upper", "left"): "upper left area of your mouth",
        ("upper", "center"): "upper middle area of your mouth",
        ("upper", "right"): "upper right area of your mouth",
        ("middle", "left"): "left side of your mouth",
        ("middle", "center"): "middle of your mouth",
        ("middle", "right"): "right side of your mouth",
        ("lower", "left"): "lower left area of your mouth",
        ("lower", "center"): "lower middle area of your mouth",
        ("lower", "right"): "lower right area of your mouth",
    }

    return location_phrases[(vertical, horizontal)]


def calculate_metric_summary(probability, result, prob1, prob2, prob3):
    """Internal model metrics (accuracy/precision/F1) — kept for engineering use,
    not shown directly to patients."""

    if probability is None:
        return {"accuracy": "N/A", "precision": "N/A", "f1_score": "N/A"}

    probability = float(min(1.0, max(0.0, probability)))

    is_negative_prediction = "No Cancer" in result or "Stage 0" in result
    class_probability = probability if is_negative_prediction else 1.0 - probability

    if is_negative_prediction:
        per_model_confidence = [prob1, prob2, prob3]
    else:
        per_model_confidence = [1.0 - prob1, 1.0 - prob2, 1.0 - prob3]

    precision_score = sum(per_model_confidence) / len(per_model_confidence)
    accuracy_score = class_probability

    if precision_score + accuracy_score <= 0:
        f1_score = 0.0
    else:
        f1_score = (2 * precision_score * accuracy_score) / (precision_score + accuracy_score)

    return {
        "accuracy": round(accuracy_score * 100, 1),
        "precision": round(precision_score * 100, 1),
        "f1_score": round(f1_score * 100, 1),
    }


def get_patient_summary(result, lesion_percent):
    """Translates the raw model prediction into plain language for the UI."""

    is_negative = "No Cancer" in result or "Stage 0" in result

    if is_negative:
        return {
            "risk_level": "low",
            "headline": "No concerning areas found",
            "message": (
                "This scan did not find anything that looks concerning. "
                "Keep up with regular dental checkups."
            ),
            "next_step": "No urgent action needed. Continue routine dental visits.",
            "icon": "check",
        }

    # Any positive/cancer-suggestive result, regardless of stage number
    if lesion_percent is not None and lesion_percent >= 15:
        severity_note = "The affected area is fairly large."
    else:
        severity_note = ""

    return {
        "risk_level": "high",
        "headline": "High risk area found",
        "message": (
            "This scan found a change in your mouth tissue that looks concerning. "
            "It's not a diagnosis — only a doctor can tell you what it actually is. "
            f"{severity_note}"
        ).strip(),
        "next_step": "See a dentist or doctor soon so they can take a closer look.",
        "icon": "alert-triangle",
    }


def get_trust_sentence(metrics):
    """Turns accuracy into one sentence a non-technical person can understand."""

    if metrics["accuracy"] == "N/A":
        return "Confidence score unavailable for this scan."

    accuracy = metrics["accuracy"]
    return f"This tool gets it right about {round(accuracy)} times out of 100 in testing. It's a helper, not a doctor."


def generate_report(
    result,
    prob1,
    prob2,
    prob3,
    lesion_area,
    lesion_percent,
    bbox,
    probability,
):
    metrics = calculate_metric_summary(probability, result, prob1, prob2, prob3)
    summary = get_patient_summary(result, lesion_percent)
    location_plain = get_lesion_location(bbox)

    report = {
        # Patient-facing fields (what the simplified UI displays)
        "risk_level": summary["risk_level"],
        "headline": summary["headline"],
        "message": summary["message"],
        "next_step": summary["next_step"],
        "icon": summary["icon"],
        "location_plain": location_plain,
        "trust_sentence": get_trust_sentence(metrics),
        "disclaimer": (
            "For research and education only. Not a medical diagnosis. "
            "Please see a qualified oral health professional."
        ),

        # Raw/technical fields (kept for logs, debugging, or an "advanced view")
        "prediction": result,
        "lesion_area": round(lesion_area, 2),
        "lesion_percent": round(lesion_percent, 2),
        "metrics": metrics,
    }

    return report