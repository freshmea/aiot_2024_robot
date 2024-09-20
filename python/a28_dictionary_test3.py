def main():
    character = {
        "name": "기사",
        "level": 12,
        "items": {
            "sword": "불꽃의 검",
            "armor": "풀플레이트",
        },
        "skill": ["베기", "세게 베기", "아주 세게 베기"]
    }

    for key, value in character.items():
        if isinstance(character[key], dict):
            for key in value:
                print(f"{key} : {value[key]}")
        elif isinstance(character[key], list):
            for ele in value:
                print(f"{key} : {ele}")
        else:
            print(f"{key} : {value}")

if __name__ == "__main__":
    main()
