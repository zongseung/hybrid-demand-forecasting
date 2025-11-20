"""
Seq2Seq LSTM Model for Residual Forecasting
Univariate time series forecasting with encoder-decoder architecture
"""

import torch
import torch.nn as nn
import numpy as np


class UnivariateSeq2SeqLSTM(nn.Module):
    """
    Univariate Encoder-Decoder LSTM for time series forecasting
    
    Args:
        hidden_size: Number of features in the hidden state
        num_layers: Number of recurrent layers
        dropout: Dropout probability
        output_size: Forecast horizon (e.g., 24 for 24 hours)
        bidirectional: If True, use bidirectional LSTM
        use_attention: If True, use attention mechanism
    """
    
    def __init__(self, 
                 hidden_size=128, 
                 num_layers=2, 
                 dropout=0.2,
                 output_size=24,
                 bidirectional=False,
                 use_attention=False):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        num_directions = 2 if bidirectional else 1
        
        # Encoder (input_size=1 for univariate)
        self.encoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Decoder (input_size=1, output_size=1)
        self.decoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size * num_directions,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention
        if use_attention:
            self.attention = nn.Linear(hidden_size * num_directions * 2, 1)
        
        # Output layer
        self.fc = nn.Linear(hidden_size * num_directions, 1)
        
    def forward(self, x, target=None, teacher_forcing_ratio=0.0):
        """
        Forward pass
        
        Args:
            x: (batch, window_size, 1) - input sequence
            target: (batch, output_size, 1) - target sequence (for teacher forcing)
            teacher_forcing_ratio: probability of using teacher forcing
            
        Returns:
            predictions: (batch, output_size, 1)
        """
        batch_size = x.size(0)
        
        # Encoder
        encoder_outputs, (hidden, cell) = self.encoder(x)
        
        if self.bidirectional:
            hidden = self._cat_directions(hidden)
            cell = self._cat_directions(cell)
        
        # Decoder initial input (zeros)
        decoder_input = torch.zeros(batch_size, 1, 1).to(x.device)
        
        outputs = []
        for t in range(self.output_size):
            if self.use_attention:
                decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
                
                # Attention mechanism
                decoder_repeated = decoder_output.repeat(1, encoder_outputs.size(1), 1)
                combined = torch.cat([decoder_repeated, encoder_outputs], dim=2)
                attention_scores = self.attention(combined).squeeze(2)
                attention_weights = torch.softmax(attention_scores, dim=1)
                
                context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
                combined_output = decoder_output + context
                prediction = self.fc(combined_output)
            else:
                decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
                prediction = self.fc(decoder_output)
            
            outputs.append(prediction)
            
            # Teacher forcing: use target or prediction as next input
            if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = target[:, t:t+1, :]
            else:
                decoder_input = prediction
        
        return torch.cat(outputs, dim=1)
    
    def _cat_directions(self, h):
        """Concatenate forward and backward hidden states"""
        return torch.cat([h[0::2], h[1::2]], dim=2)


class EarlyStopping:
    """Early stopping callback"""
    
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_model(model, train_loader, val_loader, config, device='cuda', verbose=True):
    """
    Train Seq2Seq LSTM model
    
    Args:
        model: Seq2Seq LSTM model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        config: Dictionary with training configuration
        device: Device to use ('cuda' or 'cpu')
        verbose: If True, print training progress
        
    Returns:
        best_val_loss: Best validation loss achieved
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.MSELoss()
    teacher_forcing_ratio = config.get('teacher_forcing_ratio', 0.0)
    
    # Optimizer
    if config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.0)
        )
    elif config['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.0)
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=config['learning_rate'],
            momentum=0.9,
            weight_decay=config.get('weight_decay', 0.0)
        )
    
    # Scheduler
    scheduler = None
    if config.get('scheduler') == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif config.get('scheduler') == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs']
        )
    elif config.get('scheduler') == 'reduce':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
    
    early_stopping = EarlyStopping(patience=config.get('early_stopping_patience', 10))
    best_val_loss = float('inf')
    best_model_state = None
    
    if verbose:
        print(f"\n{'Epoch':>5} {'Train Loss':>12} {'Val Loss':>12} {'Best Val':>12} {'Status':>10}")
        print("-" * 65)
    
    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_x, target=batch_y, teacher_forcing_ratio=teacher_forcing_ratio)
            loss = criterion(predictions, batch_y)
            loss.backward()
            
            if config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                predictions = model(batch_x, target=None, teacher_forcing_ratio=0.0)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Update best
        status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            status = "✓ Best"
        
        if verbose:
            print(f"{epoch+1:5d} {train_loss:12.6f} {val_loss:12.6f} {best_val_loss:12.6f} {status:>10}")
        
        # Scheduler step
        if scheduler is not None:
            if config.get('scheduler') == 'reduce':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return best_val_loss, model

