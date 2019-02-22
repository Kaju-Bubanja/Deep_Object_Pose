from tensorboardX import SummaryWriter


writer = SummaryWriter()
f = open("train_Box/loss_train.csv", 'r')
f2 = open("train_Box/loss_test.csv", 'r')
def write_out_loss(f, name):
	next(f)
	for line in f:
		splits = line.split(",")
		epoch = splits[0]
		batchid = splits[1]
		loss = splits[2]
		iter = (int(epoch)-1)*625 + int(batchid)
		writer.add_scalar(name, float(loss), iter)
write_out_loss(f, "train_loss")
write_out_loss(f2, "val_loss")
writer.close()