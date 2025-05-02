# pause and resume fifa using python when i press p it pauses the process itself and when i press r it resumes the process

import ctypes
import psutil
import keyboard
import time
import win32gui
import win32process
import win32api
import win32con
from ctypes import wintypes

# Constants for Windows API
TH32CS_SNAPTHREAD = 0x00000004
THREAD_SUSPEND_RESUME = 0x0002

# Struct for thread enumeration
class THREADENTRY32(ctypes.Structure):
    _fields_ = [
        ('dwSize', wintypes.DWORD),
        ('cntUsage', wintypes.DWORD),
        ('th32ThreadID', wintypes.DWORD),
        ('th32OwnerProcessID', wintypes.DWORD),
        ('tpBasePri', wintypes.LONG),
        ('tpDeltaPri', wintypes.LONG),
        ('dwFlags', wintypes.DWORD)
    ]

def pause_process(process_name = "FC25.exe"):
    """Pause a process by suspending all its threads."""
    # Find the process by name
    print(f"Searching for process: {process_name}")
    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'] == process_name:
            pid = proc.info['pid']
            print(f"Found {process_name} with PID: {pid}")
            
            # Create a snapshot of all threads
            h_snapshot = ctypes.windll.kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)
            if h_snapshot == -1:
                print("Failed to create snapshot")
                return
            
            # Set up the thread entry structure
            thread_entry = THREADENTRY32()
            thread_entry.dwSize = ctypes.sizeof(THREADENTRY32)
            
            # Get the first thread
            ret = ctypes.windll.kernel32.Thread32First(h_snapshot, ctypes.byref(thread_entry))
            if ret == 0:
                print("Failed to get first thread")
                ctypes.windll.kernel32.CloseHandle(h_snapshot)
                return
            
            # Iterate through all threads
            suspended_count = 0
            while ret:
                # If thread belongs to our process
                if thread_entry.th32OwnerProcessID == pid:
                    # Open the thread
                    h_thread = ctypes.windll.kernel32.OpenThread(THREAD_SUSPEND_RESUME, False, thread_entry.th32ThreadID)
                    if h_thread:
                        # Suspend the thread
                        result = ctypes.windll.kernel32.SuspendThread(h_thread)
                        if result != -1:
                            suspended_count += 1
                        # Close the thread handle
                        ctypes.windll.kernel32.CloseHandle(h_thread)
                
                # Get the next thread
                ret = ctypes.windll.kernel32.Thread32Next(h_snapshot, ctypes.byref(thread_entry))
            
            # Close the snapshot handle
            ctypes.windll.kernel32.CloseHandle(h_snapshot)
            print(f"Suspended {suspended_count} threads in {process_name} (PID: {pid})")
            return
    
    print(f"Process {process_name} not found")

def resume_process(process_name = "FC25.exe"):
    """Resume a process by resuming all its threads."""
    # Find the process by name
    print(f"Resuming {process_name}...")
    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'] == process_name:
            pid = proc.info['pid']
            print(f"Found {process_name} with PID: {pid}")
            
            # Create a snapshot of all threads
            h_snapshot = ctypes.windll.kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)
            if h_snapshot == -1:
                print("Failed to create snapshot")
                return
            
            # Set up the thread entry structure
            thread_entry = THREADENTRY32()
            thread_entry.dwSize = ctypes.sizeof(THREADENTRY32)
            
            # Get the first thread
            ret = ctypes.windll.kernel32.Thread32First(h_snapshot, ctypes.byref(thread_entry))
            if ret == 0:
                print("Failed to get first thread")
                ctypes.windll.kernel32.CloseHandle(h_snapshot)
                return
            
            # Iterate through all threads
            resumed_count = 0
            while ret:
                # If thread belongs to our process
                if thread_entry.th32OwnerProcessID == pid:
                    # Open the thread
                    h_thread = ctypes.windll.kernel32.OpenThread(THREAD_SUSPEND_RESUME, False, thread_entry.th32ThreadID)
                    if h_thread:
                        # Resume the thread
                        while True:
                            result = ctypes.windll.kernel32.ResumeThread(h_thread)
                            if result == 0:  # No more suspensions
                                break
                            resumed_count += 1
                        # Close the thread handle
                        ctypes.windll.kernel32.CloseHandle(h_thread)
                
                # Get the next thread
                ret = ctypes.windll.kernel32.Thread32Next(h_snapshot, ctypes.byref(thread_entry))
            
            # Close the snapshot handle
            ctypes.windll.kernel32.CloseHandle(h_snapshot)
            print(f"Resumed {resumed_count} thread suspensions in {process_name} (PID: {pid})")
            return
    
    print(f"Process {process_name} not found")

# def main():
#     process_name = "FC25.exe"  # Change this to the name of the process you want to pause/resume

#     print("Press 'b' to pause the process.")
#     print("Press 'r' to resume the process.")
#     print("Press 'q' to quit.")

#     is_paused = False
    
#     while True:
#         if keyboard.is_pressed('b') and not is_paused:
#             pause_process(process_name)
#             is_paused = True
#             time.sleep(0.5)  # Prevent multiple pauses
#         elif keyboard.is_pressed('r') and is_paused:
#             resume_process(process_name)
#             is_paused = False
#             time.sleep(0.5)  # Prevent multiple resumes
#         elif keyboard.is_pressed('q'):
#             # Make sure process is resumed before quitting
#             if is_paused:
#                 resume_process(process_name)
#             break
        
#         time.sleep(0.1)  # Reduce CPU usage

# if __name__ == "__main__":
#     main()