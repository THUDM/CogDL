# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals
import regex
import emoji
import os
from os import path
from PyInquirer import style_from_dict, Token, prompt
from PyInquirer import Validator, ValidationError


class Model_name_Validator(Validator):
    def validate(self, document):
        name = document.text

        model_name = name.replace(r"[^\w\s]", "").lower()
        model_file = path.exists("../cogdl/models/nn/" + model_name + ".py")
        if len(name) > 10:
            raise ValidationError(message="The name of the model is too long", cursor_position=len(name))
        elif len(name) < 2:
            raise ValidationError(message="The name of the model is too short", cursor_position=len(name))
        elif model_file:
            raise ValidationError(message="The model already exists in CogDL", cursor_position=len(name))


class Model_Maker:
    def __init__(self):
        print(emoji.emojize("CogDL - Adding a new model :rocket:"))
        self.root = os.path.dirname(os.path.abspath(__file__))
        self.style = style_from_dict(
            {
                Token.QuestionMark: "#E91E63 bold",
                Token.Selected: "#673AB7 bold",
                Token.Instruction: "",
                Token.Answer: "#2196f3 bold",
                Token.Question: "",
            }
        )
        self.questions = [
            {
                "type": "input",
                "name": "model_name",
                "message": "Name of your model: ",
                "validate": Model_name_Validator,
            },
            {
                "type": "list",
                "name": "model_task",
                "message": "Task of your mode:",
                "choices": [
                    "Node classification",
                    "Unsupervised node classification",
                    "Heterogeneous node classification",
                    "Link prediction",
                    "Multiplex link prediction",
                    "Graph classification",
                    "Unsupervised graph classification",
                ],
            },
            {
                "type": "list",
                "name": "model_type",
                "message": "Type of your mode:",
                "choices": ["Embedding method", "GNN method"],
            },
        ]

    def run(self):
        self.inputs = prompt(self.questions, style=self.style)
        self.create_model()

    def create_model(self):
        """
        1. Make a model file in cogdl/models/nn/
        2. Add to tests/tasks/ file
        3. Add to match.yml
        4. Remind the user to add their model to README.md
        """
        self.create_model_file()
        self.create_model_unit_test()
        self.add_model_to_list()
        self.readme_reminder()

    def create_model_file(self):
        self.inputs["model_name"] = self.inputs["model_name"].replace(r"[^\w\s]", "")
        self.model_name = self.inputs["model_name"].lower()
        template_file = open("templates/base_model.py", "r")
        model_file = open("cogdl/models/nn/%s.py" % self.model_name, "w")
        for line in template_file.readlines():
            if "@register_model" in line:
                line = line.replace("model_name", self.model_name)
            elif "class ModelName" in line:
                line = line.replace("ModelName", self.inputs["model_name"])
            elif "super" in line:
                line = line.replace("ModelName", self.inputs["model_name"])
            model_file.write(line)
        model_file.close()

        print("Created model file --- cogdl/models/nn/%s.py" % self.model_name)
        template_file.close()

    def create_model_unit_test(self):
        self.model_type = "gnn" if self.inputs["model_type"] == "GNN method" else "emb"
        self.model_task = self.inputs["model_task"].replace(" ", "_").lower()
        created_args = False
        print(self.model_task)
        test_file = open("tests/tasks/test_%s.py" % self.model_task, "r")
        lines = test_file.readlines()
        test_file.close()
        test_file = open("tests/tasks/test_%s.py" % self.model_task, "w")
        for line in lines:
            if "def test" in line and not created_args:
                line = "def add_%s_args(args):\n\treturn args\n\n" % self.model_name + line
                created_args = True
            elif "if __name__" in line:
                line = (
                    'def test_%s_args(args):\n\targs = get_default_args()\n\targs = add_%s_args(args)\n\ttask = build_task(args)\n\tret = task.train()\n\tassert ret["Acc"] > 0\n\n'
                    % (self.model_name, self.model_name)
                    + line
                )
            test_file.write(line)
        test_file.close()

        print("Added unit test in task --- tests/tasks/test_%s.py" % self.model_task)

    def add_model_to_list(self):
        found_task = False
        match_file = open("cogdl/match.yml", "r")
        lines = match_file.readlines()
        match_file.close()
        match_file = open("cogdl/match.yml", "w")

        for line in lines:
            if self.model_task + ":\n" == line:
                found_task = True
            if "dataset" in line and found_task:
                line = " " * 4 + "- %s\n" % self.model_name + line
                found_task = False
            match_file.write(line)

        print("Added model to the list --- match.yml")

    def readme_reminder(self):
        print(emoji.emojize("\nAll done! Don't forget to add your model to README.md :thumbs_up:"))


if __name__ == "__main__":
    mm = Model_Maker()
    mm.run()
