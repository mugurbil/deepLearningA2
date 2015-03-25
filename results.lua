require 'constants'

data = torch.load('classifications1.data', 'ascii')
print('Id,Category')
for i = 1, test_image_count do
    print(i..' , '..data[i])
end
