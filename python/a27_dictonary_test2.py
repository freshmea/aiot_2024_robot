def main():
    dictionary = {
        "name": "7D 건조 망고",
        "type": "당절임",
        "ingredient": ["망고", "설탕", "메타중아황산나트륨", "치자황색소"],
        "origin": "필리핀",
    }
    print(f'name: {dictionary["name"]}')
    print(f'type: {dictionary["type"]}')
    print(f'ingredient: {dictionary["ingredient"]}')
    print(f'origin: {dictionary["origin"]}')

    dictionary["name"] = "8D 건조 망고"
    print(f'name: {dictionary["name"]}')
    print(f"ingredient[1]: {dictionary['ingredient'][1]}")

    # valueerror
    dict_test = { 1: "123"}
    print(dict_test[1])

if __name__ == "__main__":
    main()
