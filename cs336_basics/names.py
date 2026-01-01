import random

ADJECTIVES = [
    "amber", "ancient", "bold", "bright", "calm", "cosmic", "daring", "dark",
    "eager", "electric", "fancy", "fierce", "gentle", "golden", "happy", "hidden",
    "icy", "iron", "jolly", "jumping", "keen", "kind", "lazy", "lunar",
    "magic", "mighty", "nano", "noble", "orange", "orbital", "polar", "proud",
    "quick", "quiet", "rapid", "royal", "silver", "swift", "tiny", "turbo",
    "ultra", "unique", "vivid", "vocal", "warm", "wild", "xenial", "young", "zesty"
]

NOUNS = [
    "alpaca", "arrow", "badger", "banana", "camel", "cobra", "dolphin", "dragon",
    "eagle", "ember", "falcon", "fossa", "gopher", "grape", "hippo", "hawk",
    "iguana", "ivy", "jackal", "jewel", "koala", "kite", "lemur", "llama",
    "mango", "moose", "narwhal", "newt", "otter", "owl", "panda", "penguin",
    "quail", "quokka", "rabbit", "raven", "salmon", "sloth", "tiger", "toucan",
    "urchin", "unicorn", "viper", "vulture", "walrus", "wolf", "xerus", "yak", "zebra"
]


def generate_name():
    adj = random.choice(ADJECTIVES)
    noun = random.choice(NOUNS)
    return f"{adj}-{noun}-{random.randint(0, 999):03d}"


if __name__ == '__main__':
    for _ in range(5):
        print(generate_name())
