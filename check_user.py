from firebase_config import FirebaseManager

def main():
    print("Checking for user 1234...")
    fb = FirebaseManager()
    user_id = "1234"
    
    try:
        exists = fb.check_user_exists(user_id)
        if exists:
            print(f"\nUser {user_id} exists in Firebase")
            # Get their history
            history = fb.get_user_history(user_id)
            print(f"Found {len(history)} interactions")
            
            # Get emotional progress
            progress = fb.analyze_progress(user_id)
            if progress:
                print("\nRecent emotional patterns:")
                for emotion in progress['recent_pattern']:
                    print(f"- {emotion}")
        else:
            print(f"\nUser {user_id} does not exist in Firebase")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 