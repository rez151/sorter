import os

class1_name = "weihnachtsmuetze"
class2_name = "stressball"
class3_name = "empty"


def create_folder(name):
    path = "./" + name

    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except OSError:
            print("Creation of the directory %s failed" % name)
        else:
            print("Successfully created the directory %s " % name)
    else:
        print("Directory already exists")


def main():
    create_folder(class1_name)
    create_folder(class2_name)
    create_folder(class3_name)


if __name__ == '__main__':
    main()
