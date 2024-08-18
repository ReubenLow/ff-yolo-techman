import speech_recognition as sr
import sys

keywords = ["cup", "teddy bear", "orange", "sports ball"]

def SpeechListener():
    # Initialize recognizer
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            # Adjust energy threshold
            print("Adjusting threshold...")
            r.adjust_for_ambient_noise(source, duration=1)
            print("Ready to listen...")

            print("Listening for 5 seconds...")
            audio = r.listen(source, timeout=5, phrase_time_limit=5)

            print("Recognizing Speech....")
            prompt = r.recognize_google(audio)
            result = prompt.lower()

            # Find requested object
            # Check if any keyword is in the recognized speech
            for keyword in keywords:
                if keyword in result:
                    detected_keyword = keyword
                    print(f"'{keyword}' detected in request.")
                    return detected_keyword
            
            

    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    except KeyboardInterrupt:
        print("\nTerminating...")
        sys.exit(0)

def main():
    while True:
        user_input = input("Enter 'l' to start listening, or 'q' to quit: ").lower()
        if user_input == 'l':
            # Call speechlistener to get keyword
            object = SpeechListener()
        elif user_input == 'q':
            print("Exiting...")
            break
        else:
            print("Invalid input. Please enter 'l' to listen or 'q' to quit.")

if __name__ == "__main__":
    main()
