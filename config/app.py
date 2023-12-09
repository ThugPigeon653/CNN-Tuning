from flask import Flask, Response
import json

app = Flask(__name__)

@app.route('/', methods=["GET"])
def get_json():
    json_file_path = "config.json"

    try:
        with open(json_file_path, 'r') as file:
            json_content = file.read()
        response = Response(json_content, content_type="application/json")
        return response
    except FileNotFoundError:
        return "Config file not found", 404
    except Exception as e:
        return str(e), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
