import os
from PIL import Image, ImageOps
from flask import Flask, send_file, jsonify, render_template, make_response, redirect, url_for
from flask_restx import Api, Resource, reqparse, fields
from werkzeug.datastructures import FileStorage

app = Flask(__name__)
api = Api(app)
parser = reqparse.RequestParser()

parser.add_argument('metric')
parser.add_argument('file', location='files',
                    type=FileStorage, required=True)
app.config["IMAGE_UPLOADS"] = "media/"

#
# @api.route('/vis')
# class upload(Resource):
#     def get(self):
#         return make_response(render_template("visualize.html"))


@api.route('/start')
class start(Resource):
    def get(self):
        return make_response(render_template("upload1.html"))


@api.route('/process')
class process(Resource):
    def post(self):
        args = parser.parse_args()
        uploaded_file = args['file']
        path = os.path.join(app.config["IMAGE_UPLOADS"], uploaded_file.filename)

        uploaded_file.save(os.path.join('static',path))
        img = Image.open(uploaded_file.stream)
        path = path.replace('\\', '/')
        return make_response(render_template("fab_vis.html", filename=url_for('static',filename=path),bleeding='1'))


# api.add_resource(upload, '/vis')
api.add_resource(start, '/start')
api.add_resource(start, '/process')

if __name__ == "__main__":
    app.run(debug=True)
