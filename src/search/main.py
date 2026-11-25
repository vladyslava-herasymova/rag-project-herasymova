import pipeline
import time

while True:
    print("\nWhat is your question? Enter an empty string to quit.")
    question = input()
    if (len(question.strip()) < 0):
        break
    print("Let me think for a bit...")
    start = time.time()
    answer, is_sure = pipeline.answer(question)
    end = time.time()
    print(f"The program was thinking for {end - start} seconds.")
    if (is_sure):
        print(answer)
    else:
        print(f"The program is unsure. Here is the answer given, but it should be treated very carefully: {answer}")