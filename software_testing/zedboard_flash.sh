yes | ./scripts/create_vitis -xsa_path ../simple_interface/vivado/simple_interface/staged_mac_bd_wrapper.xsa


#xterm -e "unbuffer ./scripts/file_transfer_vitis | awk '{print} {t=systime()} t-last>10000 && NR>1 {exit} {last=t}'"



