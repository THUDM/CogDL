import torch

class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            highest_train, highest_train_std = r.mean().item(), r.std().item()
            print(f'Highest Train: {r.mean():.4f} ± {r.std():.4f}')
            r = best_result[:, 1]
            highest_valid, highest_valid_std = r.mean().item(), r.std().item()
            print(f'Highest Valid: {r.mean():.4f} ± {r.std():.4f}')
            r = best_result[:, 2]
            final_train, final_train_std = r.mean().item(), r.std().item()
            print(f'  Final Train: {r.mean():.4f} ± {r.std():.4f}')
            r = best_result[:, 3]
            final_test, final_test_std = r.mean().item(), r.std().item()
            print(f'   Final Test: {r.mean():.4f} ± {r.std():.4f}')
            
            return {'train': round(final_train, 4)
                    , 'train_std': round(final_train_std, 4)
                    , 'valid': round(highest_valid, 4)
                    , 'valid_std': round(highest_valid_std, 4)
                    , 'test': round(final_test, 4)
                   , 'test_std': round(final_test_std, 4)
                   }
