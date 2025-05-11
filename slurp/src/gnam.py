
import polars as pl
from typing import List
import torch.nn as nn
import torch
from tqdm import tqdm

from slurp.src.module.core import OperationModule

class Gnam():
    def __init__(self, design: nn.Module):
        self._design = design
        self._loss_history = []
        

    @property
    def design(self):
        return self._design


    def fit(self, X: pl.DataFrame, y: pl.Series, epochs: int = 100, criterion = nn.MSELoss(), lr=0.01, weight_decay=0.01, gamma: float = 0.):
        
        # TODO REFACTOR
        optimizer = torch.optim.AdamW(params = self._design.parameters(), lr=lr, weight_decay=weight_decay)
        for epoch in (pbar:=tqdm(range(epochs))):
            optimizer.zero_grad()
            y_pred = self._design(X)
            loss = criterion(y_pred, torch.tensor(y, dtype=torch.float32).unsqueeze(1))

            # Apply constraint method
            # TODO Handle constraints
            loss += gamma*self._design.regularisation(X)

            loss.backward()
            optimizer.step()

            self._loss_history.append(loss.item())
            if epoch % 100 == 0:
                pbar.set_description(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")
        

    def predict(self, X: pl.DataFrame, index: List = None):
        """
        Make predictions
        If index is True concat columns index_col to the output
        """
        y_pred = self._design(X)
        y_pred = pl.DataFrame({'gnam': y_pred.detach().numpy().flatten()})

        if index:
            y_pred = pl.concat([X.select(index), y_pred], how='horizontal')
        return y_pred


    def predict_components(self, X: pl.DataFrame, index: List = None):
        """
        Return predictions of each splines
        """

        y_components = []

        def core(model, X):
            if not isinstance(model, OperationModule):
                y_components.append(model.predict(X))
            else:
                for m in model._modules.keys():
                    core(getattr(model, m), X)

        core(self._design, X)

        # Make sure that the columns names are unique
        columns_names = []
        for i, y in enumerate(y_components):
            col = y.columns[0]
            if col in columns_names:
                occurences = [g for g in columns_names if g.startswith(col)]
                col_index = len(occurences)
                col = f"{col}_{col_index}"

                # Rename the column
                y = y.rename({y.columns[0]: col})
                y_components[i] = y
            columns_names.append(col)
        y_components = pl.concat(y_components, how='horizontal')

        # Append total prediction
        y_components = pl.concat([y_components, self.predict(X)], how='horizontal')

        # Append additional columns
        if index:
            y_components = pl.concat([X.select(index), y_components], how='horizontal')
        
        return y_components


    def fit_diagnostic(self):
        """
        Return training diagnostics
        """
        pass

    def latex_design(self, compact: bool = False):
        """
        Return design formula in LaTeX format.
        If compact is True, return a compact version of the formula without parameters
        """
        return self.design.to_latex(compact=compact)