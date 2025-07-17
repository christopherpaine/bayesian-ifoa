import os
import ipykernel

connection_file = os.path.basename(ipykernel.connect.get_connection_file())
kernel_id = connection_file.split('-', 1)[1].split('.')[0]
print(kernel_id)

