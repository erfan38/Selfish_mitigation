import numpy as np

# Define the range for alpha and gamma
alpha_range = np.linspace(0.1, 0.5, 5)
gamma_range = np.linspace(0, 1, 4)

# Define time range
time_range = np.arange(11)

# Create a string to represent the table
table_str = ""

# Iterate over time range
for t in time_range:
    # Add separator line
    table_str += "=" * 70 + "\n"
    # Add time label
    table_str += f"t = {t:<2}" + " " * 18
    # Add alpha labels
    for alpha in alpha_range:
        table_str += f"alpha = {alpha:.2f}" + " " * 14
    table_str += "\n"
    
    # Iterate over gamma range
    for gamma in gamma_range:
        # Add gamma label
        table_str += f"gamma = {gamma:.2f}" + " " * 13
        # Calculate and add phi values
        for alpha in alpha_range:
            phi_value = gamma / (gamma + alpha * t)
            table_str += f"{phi_value:.5f}" + " " * 20
        table_str += "\n"

# Add final separator line
table_str += "=" * 70 + "\n"

# Print the table
print(table_str)
# Define the file name
file_name = "phi_table.txt"

# Open the file in write mode
with open(file_name, "w") as file:
    # Write the table string to the file
    file.write(table_str)

print(f"Table saved to {file_name}")
