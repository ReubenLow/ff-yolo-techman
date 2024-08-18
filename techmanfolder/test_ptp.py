async with techmanpy.connect_sct(robot_ip) as conn:
    trsct = conn.start_transaction()
    # Define the target position (X, Y, Z, RX, RY, RZ) in mm and degrees
    tcp_point_goal = [500, 200, 300, 0, 90, 0]
    # Set the speed percentage and acceleration duration
    speed_perc = 50
    acceleration_duration = 1000  # in milliseconds
    # Add the PTP motion command to the transaction
    await trsct.move_to_point_ptp(tcp_point_goal, speed_perc, acceleration_duration)
    # Submit the transaction to execute the PTP motion
    await trsct.submit()
