from flask import Flask, jsonify, request

app = Flask(__name__)

# Route de test
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Backend Flask opérationnel"
    })

# Route API
@app.route("/api", methods=["GET", "POST"])
def data():

    if request.method == "GET":
        return jsonify({
            "status": "success",
            "data": []
        })

    if request.method == "POST":
        payload = request.get_json()

        return jsonify({
            "status": "success",
            "received": payload
        }), 201

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8000,
        debug=True
    )