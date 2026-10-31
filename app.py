from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from src.predict import predict_image

app = Flask(__name__)

UPLOAD_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    return "." in filename and \
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():

    report = None
    image_path = None
    images = None
    advanced_images = None

    if request.method == "POST":

        file = request.files.get("image")

        if not file or file.filename == "":
            return render_template("index.html")

        if not allowed_file(file.filename):
            return render_template(
                "index.html",
                error="Invalid file type."
            )

        filename = secure_filename(file.filename)

        filepath = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        file.save(filepath)

        report, image_path = predict_image(filepath)

        # If prediction returned an error string (e.g., invalid / not an oral image),
        # convert it into the report dict shape expected by the template and
        # avoid showing any stale heatmap images.
        if isinstance(report, str) and report.startswith("Invalid"):
            report = {
                "risk_level": "unknown",
                "headline": "We couldn't read this photo",
                "message": "This doesn't look like a photo of the inside of a mouth. Please upload a clear, well-lit photo of the area you're concerned about.",
                "next_step": "Try uploading a different photo.",
                "icon": "photo-off",
                "location_plain": "Not applicable",
                "trust_sentence": "",
                "disclaimer": "For research and education only. Not a medical diagnosis. Please see a qualified oral health professional.",
                "prediction": report,
                "lesion_area": 0,
                "lesion_percent": 0,
                "metrics": {"accuracy": "N/A", "precision": "N/A", "f1_score": "N/A"},
            }
            images = None
        else:
            # Only the marked photo is shown to patients by default.
            images = {
                "lesion": "static/results/lesion.jpg"
            }

            # Kept available (not shown by default) for an optional "advanced view"
            # aimed at clinicians/engineers rather than patients.
            advanced_images = {
                "gradcam": "static/results/gradcam.jpg",
                "gradcampp": "static/results/gradcampp.jpg",
                "scorecam": "static/results/scorecam.jpg",
                "fusion": "static/results/fusion.jpg",
                "average": "static/results/average.jpg",
            }

    return render_template(
        "index.html",
        report=report,
        image_path=image_path,
        images=images,
        advanced_images=advanced_images
    )


if __name__ == "__main__":
    app.run(debug=True)