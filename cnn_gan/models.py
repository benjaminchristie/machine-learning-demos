from tkinter import W
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
import torchvision
from copy import deepcopy


def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    return


class Network(nn.Module):
    """
    arbitrary network that allows for overwriting of weights
    without overwriting gradients
    """

    def __init__(self, config):
        super(Network, self).__init__()
        self.config = config
        self.weights = nn.ParameterList()
        self.weights_bn = nn.ParameterList()
        for _, (name, param) in enumerate(self.config):
            if name == "conv2d":
                w = nn.Parameter(torch.ones(*param[:4]).T)
                torch.nn.init.kaiming_normal_(w)
                self.weights.append(w)
                self.weights.append(nn.Parameter(torch.zeros(param[1])))
            elif name == "convt2d":
                w = nn.Parameter(torch.ones(*param[:4]).T)
                torch.nn.init.kaiming_normal_(w)
                self.weights.append(w)
                self.weights.append(nn.Parameter(torch.zeros(param[2])))
            elif name == "linear":
                w = nn.Parameter(torch.ones(*param).T)
                torch.nn.init.kaiming_normal_(w)
                self.weights.append(w)
                self.weights.append(nn.Parameter(torch.zeros(param[1])))
            elif name == "bn":
                w = nn.Parameter(torch.ones(param[0]))
                self.weights.append(w)
                self.weights.append(nn.Parameter(torch.zeros(param[1])))
                # important: set grad false
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.weights_bn.extend([running_mean, running_var])
            elif name in [
                "tanh",
                "relu",
                "leaky_relu",
                "upsample",
                "max_pool2d",
                "sigmoid",
                "flatten",
                "reshape",
            ]:
                continue
            else:
                raise NotImplementedError()
        return

    def forward(self, x, weights=None):
        if weights is None:
            weights = self.weights
        idx = 0
        bn_idx = 0
        for name, param in self.config:
            if name == "conv2d":
                w, b = weights[idx], weights[idx + 1]
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
            elif name == "convt2d":
                w, b = weights[idx], weights[idx + 1]
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
            elif name == "linear":
                w, b = weights[idx], weights[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
            elif name == "bn":
                w, b = weights[idx], weights[idx + 1]
                running_mean, running_var = (
                    self.weights_bn[bn_idx],
                    self.weights[bn_idx + 1],
                )
                x = F.batch_norm(
                    x, running_mean, running_var, weight=w, bias=b, training=False
                )
                bn_idx += 2
                idx += 2
            elif name == "flatten":
                x = x.view(x.shape[0], -1)
            elif name == "reshape":
                x = x.view(x.size(0), *param)
            elif name == "relu":
                x = F.relu(x, inplace=param[0])
            elif name == "leaky_relu":
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name == "tanh":
                x = F.tanh(x)
            elif name == "sigmoid":
                x = F.sigmoid(x)
            elif name == "upsample":
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name == "max_pool2d":
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name == "avg_pool2d":
                x = F.avg_pool2d(x, param[0], param[1], param[2])
            else:
                raise NotImplementedError()

        assert idx == len(weights)
        assert bn_idx == len(self.weights_bn)
        return x


class MNIST_Classifier(nn.Module):
    def __init__(self, n_inputs=784, n_classes=10):
        super(MNIST_Classifier, self).__init__()

        self.fc1 = nn.Linear(n_inputs, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, n_classes)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.dropout(self.relu(self.fc4(x)))
        x = self.sigmoid(self.fc5(x))
        return x

    def predict(self, x):
        x = self.forward(x)
        return torch.multinomial(x / torch.sum(x), 1)


class MAML(nn.Module):
    """
    this variant of the MAML_Classifier accepts
    only one model as input (i.e., base theta)
    and tries to adapt to different tasks
    """

    def __init__(
        self,
        model: Network,
        outer_lr: float = 2e-3,
        inner_lr: float = 1e-2,
        update_steps: int = 5,
    ):
        super(MAML, self).__init__()
        self.network = model
        self.outer_lr = outer_lr
        self.inner_lr = inner_lr
        self.update_steps = update_steps
        self.optim = Adam(self.network.parameters(), self.outer_lr)

    def forward(self, x, weights=None):
        return self.network(x, weights)

    def step_weights(self, grad, weights):
        """theta - alpha * grad"""
        return list(map(lambda p: p[1] - self.inner_lr * p[0], zip(grad, weights)))

    def update(self, x_spt, y_spt, x_q, y_q, device, loss_func=F.cross_entropy):
        n_tasks = len(x_q)
        cum_loss_q = torch.FloatTensor([0.0]).to(device)
        for t in range(n_tasks):
            pred = self.forward(x_spt[t])
            loss = loss_func(pred, y_spt[t])
            grad = torch.autograd.grad(loss, self.network.parameters())  # type: ignore
            weights = self.step_weights(grad, self.network.parameters())

            for k in range(self.update_steps):
                pred = self.forward(x_spt[t], weights)
                loss = loss_func(pred, y_spt[t])
                grad = torch.autograd.grad(loss, weights)
                weights = self.step_weights(grad, weights)
                pred_q = self.forward(x_q[t], weights)
                loss_q = loss_func(pred_q, y_q[t])

                if k == self.update_steps - 1:
                    cum_loss_q += loss_q

        cum_loss_q /= n_tasks
        self.optim.zero_grad()
        cum_loss_q.backward()
        self.optim.step()
        return cum_loss_q.item()

    def finetune(
        self, x_spt, y_spt, x_q, y_q, device, epochs=1, loss_func=F.cross_entropy
    ):
        n_tasks = len(x_q)

        net = deepcopy(self.network)
        optim = Adam(net.parameters(), self.outer_lr)

        for _ in range(epochs):
            cum_loss_q = torch.FloatTensor([0.0]).to(device)
            for t in range(n_tasks):
                pred = net(x_spt[t])
                loss = loss_func(pred, y_spt[t])
                grad = torch.autograd.grad(loss, net.parameters())  # type: ignore
                weights = self.step_weights(grad, net.parameters())

                for k in range(self.update_steps):
                    pred = net(x_spt[t], weights)
                    loss = loss_func(pred, y_spt[t])
                    grad = torch.autograd.grad(loss, weights)
                    weights = self.step_weights(grad, weights)
                    pred_q = net(x_q[t], weights)
                    loss_q = loss_func(pred_q, y_q[t])

                    if k == self.update_steps - 1:
                        cum_loss_q += loss_q

            cum_loss_q /= n_tasks
            optim.zero_grad()
            cum_loss_q.backward()
            optim.step()
        return net


class AttentionHead(nn.Module):
    """
    implementation of a single attention head
    """

    def __init__(self, head_size, n_embd, block_size, dropout):
        super(AttentionHead, self).__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.qry = nn.Linear(n_embd, head_size, bias=False)
        self.val = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, T, _ = x.shape
        k = self.key(x)
        q = self.qry(x)
        weight = q @ k.transpose(-2, -1) * (k.shape[-1]) ** (-0.5)
        weight = weight.masked_fill(self.tril[:T, :T] == 0, -torch.inf)  # type: ignore
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)
        v = self.val(x)
        return weight @ v


class MultiAttentionHead(nn.Module):
    def __init__(self, n_heads, head_size, n_embd, block_size, dropout):
        super(MultiAttentionHead, self).__init__()
        self.heads = nn.ModuleList(
            [
                AttentionHead(head_size, n_embd, block_size, dropout)
                for _ in range(n_heads)
            ]
        )
        self.proj = nn.Linear(head_size * n_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.dropout(self.proj(x))
        return x


class FeedForward(nn.Module):
    def __init__(self, n_embd, hidden_dim, dropout):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(n_embd, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_embd)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(self.fc2(x))
        return x


class Block(nn.Module):
    """
    a single transformer block
    """

    def __init__(self, n_embd, n_head, block_size, dropout, ff_hidden_dim):
        super(Block, self).__init__()
        assert n_embd % n_head == 0, "n_embd must be divisble by n_head"

        self.head_size = n_embd // n_head
        self.attention = MultiAttentionHead(
            n_head, self.head_size, n_embd, block_size, dropout
        )
        self.linear = FeedForward(n_embd, ff_hidden_dim, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))  # x + ... adds residual encoding
        x = x + self.linear(self.ln2(x))  # x + ... adds residual encoding
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_embd,
        block_size,
        n_head,
        dropout,
        ff_hidden_dim,
        n_layers,
        device,
        lr: float = 1e-4,
    ):
        super(TransformerDecoder, self).__init__()
        self.device = device
        self.tok_embeddings = nn.Embedding(vocab_size, n_embd)
        self.pos_embeddings = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[
                Block(n_embd, n_head, block_size, dropout, ff_hidden_dim)
                for _ in range(n_layers)
            ]
        )
        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(_init_weights)

        self.optim = AdamW(self.parameters(), lr=lr)

    def forward(self, y):
        _, T = y.shape
        tok_emb = self.tok_embeddings(y)
        pos_emb = self.pos_embeddings(torch.arange(T, device=self.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.lm_head(self.ln(x))
        return x

    def sample(self, x):
        x = self.forward(x)
        return torch.multinomial(F.softmax(x[:, -1, :], dim=-1), num_samples=1)

    def update_parameters(self, x, y, update_weights=True):
        if update_weights:
            self.optim.zero_grad()
            pred = self.forward(x)
            B, T, C = pred.shape
            pred = pred.view(B * T, C)
            y = y.view(B * T)
            loss = F.cross_entropy(pred, y)
            loss.backward()
            self.optim.step()
        else:
            pred = self.forward(x)
            B, T, C = pred.shape
            pred = pred.view(B * T, C)
            y = y.view(B * T)
            loss = F.cross_entropy(pred, y)
        return loss.item()
