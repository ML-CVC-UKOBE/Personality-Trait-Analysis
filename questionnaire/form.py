from flask import Flask, render_template, redirect, url_for
from flask import request
import os
import uuid
from pymongo import MongoClient
from bson.objectid import ObjectId
import time
import argparse
import numpy as np



parser = argparse.ArgumentParser("")
parser.add_argument("passw", type=str)
args = parser.parse_args()
app = Flask(__name__)
client = MongoClient('mongodb://%s:%s@localhost' %("admin", args.passw), 27017)
db = client.personality

@app.route('/')
def index(message=""):
    id = db["sessions"].insert({})
    return render_template('%s.html' % "form", message=message, user_id=id)


@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        user_id = request.form["user_id"]
        answers = {}
        global names
        global scores
        global percent
        names = ["Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness"]
        signs = [[+1, -2, +3, -4, +5],
                 [-1, +2, -3, +4, -5],
                 [+1, -2, +3, -4, +5],
                 [-1, +2, -3, +4, -5],
                 [+1, -2, +3, -4, +5],
                 [-1, +2, -3, -4, -5],
                 [+1, -2, +3, -4, +5],
                 [-1, +2, -3, -4, +5],
                 [+1, +2, +3, -4, +5],
                 [-1, +2, +3, -4, +5]]
        mask = [[1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 0, 0, 1],
                [1, 0, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 0, 0, 1]]
        totals = [40, 45, 40, 35, 40]


        traits = ["EXT", "AGR", "CSN", "EST", "OPN"]
        scores = {trait: 0 for trait in names}

        for i, trait in enumerate(traits):
            for j in range(10):
                question = "%s%d" % (trait, 1 + j)
                if mask[j][i] == 0:
                    continue
                try:
                    answer = int(request.form[question])
                    scores[names[i]] += 100 * (answer if signs[j][i] > 0 else 6 - answer) / float(totals[i])
                except Exception as e:
                    return redirect(url_for("/"))
                answers[question] = answer
        doc = db["sessions"].find_one({u"_id": ObjectId(user_id)})
        doc["scores"] = scores
        doc["questions"] = answers
        db["sessions"].update_one({"_id": ObjectId(user_id)}, {"$set": doc})
        return show_images(user_id)
        #show_results(names, scores, percent)

def show_images(user_id):
    ret = []
    traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
    modifiers = ["Low", "High"]
    indices = np.random.permutation(len(traits))

    for i in indices:
        mod_left = np.random.randint(2)
        mod_right = 1 - mod_left
        path_left = os.path.join("static", "%s_%s" %(modifiers[mod_left], traits[i]))
        path_right = os.path.join("static", "%s_%s" %(modifiers[mod_right], traits[i]))
        img_left = os.listdir(path_left)[0]
        img_right = os.listdir(path_right)[0]
        ret.append({"left": os.path.join(path_left, img_left), "right": os.path.join(path_right, img_right), "name": traits[i], "value_left": 1 - 2 * mod_left, "value_right": 1 - 2 * mod_right})
    return render_template("images.html", user_id=user_id, items=ret)


@app.route('/imgquery', methods=['POST'])
def get_img_choice():
    if request.method == "POST":
        user_id = request.form["user_id"]
        exists = db["forms"].find_one({"_id": ObjectId(user_id)})
        if exists is None:
            doc = db["sessions"].find_one({u"_id": ObjectId(user_id)})
            traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
            ret = {trait: 0 for trait in traits}
            for trait in traits:
                ret[trait] += int(request.form[trait])

            doc["images"] = ret
            db["sessions"].update_one({"_id": ObjectId(user_id)}, {"$set": doc})
            save(user_id)

    return show_results(user_id)

def save(user_id):
    doc = db["sessions"].find_one({u"_id": ObjectId(user_id)})
    assert("scores" in doc)
    assert("images" in doc)
    assert(len(doc["questions"].values()) == 40)
    assert(len(doc["scores"]) == 5)
    assert(len(doc["images"]) == 5)
    db["forms"].insert_one(doc)


def show_results(user_id):
    doc = db["forms"].find_one({u"_id": ObjectId(user_id)})
    scores = doc["scores"]
    ret = ", ".join(["%s: %.02f%%" % (key, scores[key]) for key in scores])
    nscores = {"n%s" %key: 100 - scores[key] for key in scores}
    nscores.update(scores)
    nscores = { k: "%.02f%%" %v for k,v in nscores.items()}
    return render_template("results.html", **nscores)


if __name__ == "__main__":
    app.run(host='0.0.0.0')
