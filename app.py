from flask import Flask, render_template, request
from chatbot import predict_class, get_response, intents

# import sys, time

# def typing_print(text):          #Prints text in typewriter style
#     for character in text:
#         sys.stdout.write(character)
#         sys.stdout.flush()
#         time.sleep(0.01)

app = Flask(__name__)
app.config['SECRET_KEY'] = "I've_got_a_secret_a_secret_secret"

response_list = []

@app.route("/")
def home():
    global response_list
    response_list.clear()
    return render_template("index.html")

@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    global response_list
    if request.method == "POST":
        message = request.form['message']
        message = message.lower()
        ints = predict_class(message)
        res = get_response(ints, intents)
        response_list.append(res)
        if len(response_list) >5:
            response_list.remove(response_list[0])
        return render_template("chatbot.html", message=message, response_list=response_list)
    return render_template("chatbot.html", message="", response_list=response_list)

if __name__ == "__main__":
    app.run(debug=True)
