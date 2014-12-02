function cova = get_cov_mat()
    load landmarks_lfw.mat;
    
    % filter out non frontal faces
    bs = [landmarks.bs];
    % TODO: currently using only frontal pose
    frontal_samples = bs([bs.c]==7);
    
    % parse detected landmarks
    no_samples = size(frontal_samples,2); 
    samples = zeros(136, no_samples);
    for i=1:no_samples
        samples(:,i) = parse_detector_results(frontal_samples(i).xy);
    end
    
    cova = cov(samples');
    %cova = zeros(136,136);
    %denominator = 1/(no_samples-1);
    %for i=1:136
    %    for j=1:136
    %        mean_i = mean(samples(i,:));
    %        mean_j = mean(samples(j,:));
    %        cova(i,j)=denominator * (samples(j,:) - mean_j) * (samples(i,:) - mean_i)';
    %    end
    %end
end

