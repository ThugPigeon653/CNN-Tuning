from flask import Flask, Response
import yaml

app = Flask(__name__)

@app.route('/', methods=["GET"])
def get_yaml():
    yaml_file_path = "config.yaml"

    try:
        with open(yaml_file_path, 'r') as file:
            yaml_content = file.read()
        response = Response(yaml_content, content_type="text/yaml")
        return response
    except FileNotFoundError:
        return "YAML file not found", 404
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)