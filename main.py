"""
Web service main script
"""
import logging
import os

from flask import Flask, Response, make_response, render_template, request

from stylize import stylize_image

app = Flask(__name__)
STYLE_MODEL_PATH = "styles"
logger = logging.getLogger("styling")
logger.setLevel(logging.INFO)
formater = logging.Formatter('%(levelname)s: [%(asctime)s] %(message)s')
main_handler = logging.StreamHandler()
main_handler.setFormatter(formater)
logger.addHandler(main_handler)

if not os.path.exists(STYLE_MODEL_PATH):
    os.makedirs(STYLE_MODEL_PATH)


@app.route("/")
def index() -> Response:
    """
    Main page of the service
    :return: rendered page
    """
    return render_template("index.html")


@app.route("/stylize", methods=["POST"])
def stylize() -> Response:
    """
    Data stylization route
    :return: stylized image bytes as response
    """
    style = request.values["style"]
    content = request.values["content"]
    content = content[content.index(",") + 1:]
    logger.info("Got file and style %s" % style)
    img_bytes = stylize_image(content, style)
    img_bytes = f"data:image/jpeg;base64,{img_bytes}"
    logger.info("Result send back")
    return make_response(img_bytes, 200)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
