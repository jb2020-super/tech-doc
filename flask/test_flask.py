from flask import Flask, request

app = Flask(__name__)

@app.route('/test', methods=['POST'])
def my_method():
    name = request.json['name']
    year = request.json['year']
    return name + ' was born in ' + str(year) + '.'

if __name__=='__main__':
    app.debug = True
    app.run()