import random


class BackgoundGenerator:
    def __init__(self):
        self.backgrounds = [
            "Wallpaper",
            "In forest",
            "Beach",
            "In the desert",
            "In the city",
            "On the mountain",
            "In the space",
            "Underwater",
        ]

    def generate(self):
        background = random.choice(self.backgrounds)
        return f"""Background:{background}\n"""


class StyleGenerator:
    def __init__(self):
        self.styles = [
            "Cartoon",
            "Japan-animation",
            "Pencil sketch",
            "Watercolor",
            "Oil painting",
            "3D render",
            "Disney style",
            "Studio Ghibli style",
        ]

    def generate(self):
        style = random.choice(self.styles)
        return f"""Style:{style}\n"""


class RaceGenerator:
    def __init__(self):
        self.skin_tones = ["Light", "Medium", "Pale", "White"]
        self.races = [
            "Caucasian",
            "East Asian",
            "Southeast Asian",
            "African",
            "Hispanic",
        ]

    def generate(self):
        skin_tone = random.choice(self.skin_tones)
        race = random.choice(self.races)
        return f"""Skin Tone:{skin_tone}\nRace:{race}\n"""


class GenderGenerator:
    def __init__(self):
        self.genders = ["Male", "Female"]

    def generate(self):
        gender = random.choice(self.genders)
        return f"""Gender:{gender}\n"""


class AgeGenerator:
    def __init__(self):
        self.age_groups = ["Child", "Teenager", "Adult", "Senior"]

    def generate(self):
        age_group = random.choice(self.age_groups)
        return f"""Age group:{age_group}\n"""


class CharacterGenerator:
    def __init__(self):
        self.fattness_levels = ["Slim", "Average", "Athletic", "Chubby", "Obese"]

    def generate(self):
        fattness = random.choice(self.fattness_levels)
        return f"""Fattness:{fattness}\n"""


class MoodGenerator:
    def __init__(self):
        self.extras = [
            "Handsome",
            "Beautiful",
            "Cute",
            "Ugly",
            "Attractive",
            "Unattractive",
            "Charming",
            "Dull",
            "Mysterious",
        ]

    def generate(self):
        extra = random.choice(self.extras)
        return f"""Mood:{extra}\n"""


class AccessaryGenerator:
    def __init__(self):
        self.accessaries = [
            "Earrings",
            "Necklace",
            "Bracelet",
            "Ring",
            "Hat",
            "Scarf",
            "None",
            "Face-mask",
            "Sunglasses",
            "Glasses",
            "Microphone",
            "Headphones",
        ]

    def generate(self):
        accessary = random.choice(self.accessaries)

        return f"""Accessary:{accessary}\n"""


class ActionGenerator:
    def __init__(self):
        self.actions = [
            "Smiling",
            "Frowning",
            "Laughing",
            "Crying",
            "Shouting",
            "Thinking",
            "Surprised",
            "Angry",
            "Calm",
            "Hold a something",
        ]

    def generate(self):
        action = random.choice(self.actions)
        return f"""Action:{action}\n"""


class HairGenerator:
    def __init__(self):
        self.hair_styles = [
            "Straight",
            "Wavy",
            "Curly",
            "Ponytail",
            "Braided",
            "Buzz cut",
            "Afro",
            "Mohawk",
            "Hat or Headwear",
        ]

    def generate(self):
        is_bald = random.choices([True, False], weights=[0.05, 0.95], k=1)[0]
        if is_bald:
            return """Hair Style:Bald\n"""
        else:
            hair_style = random.choice(self.hair_styles)
            return f"""Hair Style:{hair_style}\n"""


class EyeGenerator:
    def __init__(self):
        self.eye_colors = ["Brown", "Blue", "Green", "Amber"]

    def generate(self):
        eye_color = random.choice(self.eye_colors)
        return f"""Eye color:{eye_color}\n"""


class PromptGenerator:
    def __init__(self):
        self.background_generator = BackgoundGenerator()
        self.style_generator = StyleGenerator()
        self.race_generator = RaceGenerator()
        self.gender_generator = GenderGenerator()
        self.age_generator = AgeGenerator()
        self.character_generator = CharacterGenerator()
        self.mood_generator = MoodGenerator()
        self.accessary_generator = AccessaryGenerator()
        self.action_generator = ActionGenerator()
        self.hair_generator = HairGenerator()
        self.eye_generator = EyeGenerator()

    def generate(self):
        face_description = ""
        face_description += self.background_generator.generate()
        face_description += self.style_generator.generate()
        face_description += self.race_generator.generate()
        face_description += self.gender_generator.generate()
        face_description += self.age_generator.generate()
        face_description += self.character_generator.generate()
        face_description += self.mood_generator.generate()
        face_description += self.accessary_generator.generate()
        face_description += self.action_generator.generate()
        face_description += self.hair_generator.generate()
        face_description += self.eye_generator.generate()
        return face_description
