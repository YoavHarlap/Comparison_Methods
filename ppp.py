import difflib
import sys

def text_compare(text1, text2):
    sm = difflib.SequenceMatcher(None, text1, text2)
    diff_lines = sm.get_opcodes()

    print("Differences:")
    for opcode in diff_lines:
        if opcode[0] == 'equal':
            print(f"\033[92m {opcode[1]}: {''.join(text1[opcode[1]:opcode[2]])}\033[0m")
        elif opcode[0] == 'replace':
            print(f"\033[91m {opcode[1]}: {''.join(text1[opcode[1]:opcode[2]])} -> {''.join(text2[opcode[1]:opcode[2]])}\033[0m")
        elif opcode[0] == 'insert':
            print(f"\033[94m {opcode[1]}: +{''.join(text2[opcode[1]:opcode[2]])}\033[0m")
        elif opcode[0] == 'delete':
            print(f"\033[91m {opcode[1]}: -{''.join(text1[opcode[1]:opcode[2]])}\033[0m")

if __name__ == "__main__":
    text1 = input("Enter the first text: ")
    text2 = input("Enter the second text: ")

    text_compare(text1, text2)
    import winsound

    winsound.Beep(2500, 1000)  # Frequency: 2500 Hz, Duration: 1000 ms