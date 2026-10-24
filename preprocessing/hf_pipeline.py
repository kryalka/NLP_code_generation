from preprocessing.preprocessing_code import main as clean_code
from preprocessing.preprocessing_questions import main as clean_questions
from preprocessing.split import main as create_splits
from preprocessing.body_only_hf import main as create_body_dataset
from preprocessing.upload_tokens import main as create_wrapper_tokens


def main():
    clean_questions()
    clean_code()
    create_splits()
    create_body_dataset()
    create_wrapper_tokens()


if __name__ == "__main__":
    main()
