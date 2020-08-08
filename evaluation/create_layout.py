import torch

######################################
###################################
######################################

imgs = imgs[0, :].view(1, 3, 64, 64)

objs = torch.LongTensor([6, 4, 6, 4, 11, 17, 119]).to(device)

boxes = torch.FloatTensor([[0.1, 0.3, 0.25, 0.9],
                           [0.1, 0.4, 0.25, 0.62],
                           [0.7, 0.3, 0.8, 0.9],
                           [0.7, 0.4, 0.8, 0.62],
                           [0, 0, 1, 0.4],
                           [0, 0.4, 1, 1],
                           [0, 0.3, 1, 0.5]]).to(device)

H, W = 64, 64
masks = torch.zeros(7, 1, H, W).to(device)

# obj_idx_mapping = {}
for u in range(6):
    x0, y0, x1, y1 = boxes[u].cpu().numpy().tolist()
    masks[u, :, round(y0 * H):round(y1 * H), round(x0 * W):round(x1 * W)] = 1

obj_to_img = torch.Tensor([0, 0, 0, 0, 0, 0, 0])

z = torch.randn(7, 64).to(device)

attribute = torch.zeros(7, 106)
attribute[1, 8] = 1
attribute[3, 94] = 1
attribute = attribute.to(device)

masks_shift = torch.ones(7, 1, 64, 64).to(device)

boxes_shift = torch.rand(7, 4).to(device)

attribute_est = torch.ones(7, 106).to(device)

imgs, objs, boxes, masks, obj_to_img, z, attribute, masks_shift, boxes_shift = \
    imgs.to(device), objs.to(device), boxes.to(device), masks.to(device), \
    obj_to_img, z.to(device), attribute.to(device), masks_shift.to(device), boxes_shift.to(device)

###################################################
##########################################
##############################################


