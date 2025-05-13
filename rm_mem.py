# from rl import DQNAgent

# model_name = "MODEL_5_11.pth"

# if __name__ == "__main__":
#     # Create an instance of the DQNAgent
#     agent = DQNAgent(model_name)

    
#     for i in range(200):
#         agent.replay(64)


#     # agent.delete_specific_memory(941)

#     agent.save(model_name)



from rl import DQNAgent

model_name = "MODEL_5_11.pth"

if __name__ == "__main__":
    # Create an instance of the DQNAgent
    agent = DQNAgent(model_name)
    
    # Train the model
    for i in range(300):
        agent.replay(128)
        
        # Optional: Print progress every 50 iterations
        if i % 50 == 0:
            print(f"Training progress: {i}/300 iterations completed")

    agent.save(model_name)