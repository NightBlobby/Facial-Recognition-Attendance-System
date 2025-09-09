import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=int, required=True, help='0=Enroll, 1=Attendance')
    args = parser.parse_args()
    print(f'Running mode {args.mode}')
