function theta_prime = calculate_theta_prime(theta_rad)
    % Convert degrees to radians
    %theta_rad = deg2rad(theta);
    
    % Given formula: theta_prime = atan(1/6 * tan(theta))
    theta_prime = atan(1/sqrt (3) * tan(theta_rad));
    
    % Convert radians back to degrees
    %theta_prime = rad2deg(theta_prime);
end

