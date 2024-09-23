
def flatten(data):
    output = []
    for li in data:
        if isinstance(li, list):
            output.extend(flatten(li))
        else:
            output.append(li)
    return output

def main():
    example=[1,2,3,[4,[5,6]],[7,[8,9]]]
    print(flatten(example))

if __name__ == "__main__":
    main()
