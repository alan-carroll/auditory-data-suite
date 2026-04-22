import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import warnings

_NORM_SIZE = (48, 32)

@dataclass
class OCRResult:
    """Single number recognition outcome."""
    crop: np.ndarray
    number: Optional[int] = None
    confidence: float = 0.0
    metadata: dict = field(default_factory=dict)


class DigitOCR:
    """
    Digit recognition via exact-match vectorized template matching.
    Optimized strictly for pure digital text on solid backgrounds.
    """
    def __init__(self):
        self._working_templates = {d: None for d in range(10)}
        self.templates_3d = None
        self.template_digits = None

    # ────────────────────────────────────────────────────────────
    #  Construction
    # ────────────────────────────────────────────────────────────

    @classmethod
    def from_font(cls, font_path, font_size=60):
        from PIL import Image, ImageDraw, ImageFont

        instance = cls()
        font = ImageFont.truetype(str(font_path), font_size)
        for digit in range(10):
            canvas = Image.new("L", (font_size * 2, font_size * 2), 255)
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (font_size // 2, font_size // 4),
                str(digit), fill=0, font=font,
            )
            arr = np.array(canvas)
            _, binary = cv2.threshold(arr, 128, 255, cv2.THRESH_BINARY_INV)
            instance._working_templates[digit] = cls._normalize(binary)
        instance.validate()
        instance._compile_templates()
        return instance

    @classmethod
    def load(cls, path):
        instance = cls()
        data = np.load(str(path))
        for d in range(10):
            key = f"digit_{d}"
            if key in data:
                instance._working_templates[d] = data[key]
        instance.validate()
        instance._compile_templates()
        return instance

    @classmethod
    def bootstrap(cls, number_crops, required_conf=0.85):
        """
        If you don't have a font file, create templates from input images.
        
        Attempts to match against any existing templates first.
        Any digit that doesn't match with at least required_conf receives
        a user-input label and becomes that digit's template.
        """
        import matplotlib.pyplot as plt

        instance = cls()
        all_digits = []
        for crop in number_crops:
            blobs = cls._extract_blobs(crop)
            all_digits.extend(blob for blob, _ in blobs)

        labeled_count = 0
        match_count = 0
        print(f"Extracted {len(all_digits)} individual digit images.")
        print("Label each unique digit:  0-9 = digit,  s = skip\n")

        for digit_binary in all_digits:
            normed = cls._normalize(digit_binary)

            if any(t is not None for t in instance._working_templates.values()):
                _, best_score = instance._match_single(normed)
                if best_score >= required_conf:
                    match_count += 1
                    continue

            display = cv2.resize(
                255 - digit_binary, (0, 0),
                fx=4, fy=4, interpolation=cv2.INTER_NEAREST,
            )
            fig, ax = plt.subplots(figsize=(2.5, 3))
            ax.imshow(display, cmap="gray", vmin=0, vmax=255)
            ax.set_title("What digit?")
            ax.axis("off")
            fig.tight_layout()
            plt.show(block=False)
            fig.canvas.draw()
            fig.canvas.flush_events()

            while not ((resp := input("  Label (0-9 / s): ")) == "s" 
                       or (resp.isdigit() and 0 <= int(resp) <= 9)):
                continue
            plt.close(fig)

            if resp == "s":
                continue
            if resp.isdigit() and 0 <= int(resp) <= 9:
                instance._working_templates[int(resp)] = normed
                labeled_count += 1
                instance._compile_templates()

        instance.validate()
        instance._compile_templates()
        print(f"\nDone: {labeled_count} labeled, {match_count} template-matched.")
        instance.print_summary()
        return instance

    # ────────────────────────────────────────────────────────────
    #  Persistence
    # ────────────────────────────────────────────────────────────

    def save(self, path):
        data = {}
        for d in range(10):
            if self._working_templates[d] is not None:
                data[f"digit_{d}"] = self._working_templates[d]
        np.savez_compressed(str(path), **data)
        print(f"Saved templates to {path}")

    @staticmethod
    def export_results_csv(results, path):
        """Export results to CSV: number,x,y"""
        path = Path(path)
        with path.open("w") as f:
            f.write("number,x,y\n")
            for r in results:
                x = r.metadata.get("x", "")
                y = r.metadata.get("y", "")
                f.write(f"{r.number},{x},{y}\n")
        print(f"Exported {len(results)} rows to {path}")


    # ────────────────────────────────────────────────────────────
    #  Template inspection
    # ────────────────────────────────────────────────────────────

    def preview_templates(self):
        """Display a preview of all templates."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 10, figsize=(12, 2), squeeze=False)
        fig.suptitle("Compiled Master Templates", fontsize=11, fontweight="bold")

        for d in range(10):
            ax = axes[0, d]
            ax.set_xticks([])
            ax.set_yticks([])
            
            if self._working_templates[d] is None:
                ax.axis("off")
                ax.text(0.5, 0.5, "✗", ha="center", va="center",
                        fontsize=20, color="#dd3333", transform=ax.transAxes)
                ax.set_title(f"{d}", fontsize=10, fontweight="bold", color="#dd3333")
            else:
                ax.imshow(255 - self._working_templates[d], cmap="gray", vmin=0, vmax=255)
                ax.set_title(f"{d}", fontsize=10, fontweight="bold", color="#22aa22")

        plt.tight_layout()
        plt.show(block=True)

    def validate(self):
        """Verify complete template set is loaded."""
        missing = [d for d in range(10) if self._working_templates[d] is None]
        if missing:
            msg = "Template set is incomplete:\n" + "".join(
                [f"  {d}: ✗ MISSING\n" for d in missing]
            )
            warnings.warn(msg, UserWarning)
            warnings.warn(
                "Incomplete template sets exclude missing digits from " \
                "matching; inputs containing those digits may be " \
                "misclassified as the closest available template.",
                UserWarning
            )
            return False
        return True

    def print_summary(self):
        for d in range(10):
            mark = "✓" if self._working_templates[d] is not None else "✗"
            print(f"  [{mark}] {d}")

    # ────────────────────────────────────────────────────────────
    #  Recognition
    # ────────────────────────────────────────────────────────────

    def recognize(self, gray_image, metadata=None):
        res = OCRResult(crop=gray_image, metadata=dict(metadata or {}))
        blobs = self._extract_blobs(gray_image)
        if not blobs:
            return res

        digits = []
        confidences = []
        for digit_binary, _ in blobs:
            normed = self._normalize(digit_binary)
            d, score = self._match_single(normed)
            if d is None:
                return res
            digits.append(d)
            confidences.append(score)

        res.number = int("".join(str(d) for d in digits))
        res.confidence = sum(confidences) / len(confidences)
        
        return res

    # ────────────────────────────────────────────────────────────
    #  Visual review
    # ────────────────────────────────────────────────────────────

    def review_results(
        self,
        results,
        cols=10,
        confidence_threshold=0.85,
        save_proof_sheet=None,
    ):
        """
        Visual proof sheet.

        Args:
            results: OCRResult's returned from recognize
            cols: grid columns (default 10 for easy row tracking)
            save_proof_sheet: if given, saves proof sheet PNG instead of display
        """
        import matplotlib.pyplot as plt
        from collections import Counter

        # ── Analysis ──
        numbers = [r.number for r in results if r.number is not None]
        counts = Counter(numbers)
        duplicates = {n for n, c in counts.items() if c > 1}

        STATUS_COLORS = {
            "ok":        "#22aa22",
            "low_conf":  "#cc9900",
            "duplicate": "#dd6600",
            "failed":    "#dd2222",
        }

        def status_of(r):
            if r.number is None:
                return "failed"
            if r.number in duplicates:
                return "duplicate"
            if r.confidence < confidence_threshold:
                return "low_conf"
            return "ok"

        # ── Proof Sheet ──
        sorted_res = sorted(
            results, key=lambda r: (r.number is None, r.number or 0),
        )
        n = len(sorted_res)
        rows = max(1, -(-n // cols))

        fig, axes = plt.subplots(
            rows, cols, figsize=(cols * 1.3, rows * 1.6), squeeze=False,
        )
        fig.suptitle(
            f"Proof Sheet — {n} numbers",
            fontsize=11, fontweight="bold",
        )

        for idx in range(rows * cols):
            r_i, c_i = divmod(idx, cols)
            ax = axes[r_i, c_i]
            ax.set_xticks([])
            ax.set_yticks([])

            if idx >= n:
                ax.axis("off")
                continue

            res = sorted_res[idx]
            s = status_of(res)
            color = STATUS_COLORS[s]

            ax.imshow(res.crop, cmap="gray", vmin=0, vmax=255)
            label = str(res.number) if res.number is not None else "???"
            conf_str = f"{res.confidence:.0%}" if res.number is not None else ""
            ax.set_title(f"{label}\n{conf_str}", fontsize=7,
                         color=color, fontweight="bold")

            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2.5)
                spine.set_visible(True)

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)

        if save_proof_sheet:
            fig.savefig(str(save_proof_sheet), dpi=150, bbox_inches="tight")
            print(f"Proof sheet saved to {save_proof_sheet}")
            plt.close(fig)
        else:
            plt.show(block=True)

    # ────────────────────────────────────────────────────────────
    #  Internals
    # ────────────────────────────────────────────────────────────

    @classmethod
    def _normalize(cls, binary):
        coords = np.where(binary > 0)
        if len(coords[0]) == 0:
            return np.zeros(_NORM_SIZE, dtype=np.uint8)

        y0, y1 = coords[0].min(), coords[0].max() + 1
        x0, x1 = coords[1].min(), coords[1].max() + 1
        cropped = binary[y0:y1, x0:x1]

        th, tw = _NORM_SIZE
        h, w = cropped.shape
        scale = min((th - 4) / h, (tw - 4) / w)
        nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
        resized = cv2.resize(cropped, (nw, nh), interpolation=cv2.INTER_AREA)
        _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

        canvas = np.zeros(_NORM_SIZE, dtype=np.uint8)
        yo, xo = (th - nh) // 2, (tw - nw) // 2
        canvas[yo:yo + nh, xo:xo + nw] = resized
        return canvas

    @staticmethod
    def _extract_blobs(gray_image):
        _, binary = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours]
        # Small min. h,w should ignore any random pixel artifacts
        if boxes:
            boxes = [(x, y, w, h) for x, y, w, h in boxes if h > 10 and w > 2]
        boxes.sort(key=lambda b: b[0])
        return [(binary[y:y+h, x:x+w], (x, y, w, h)) for x, y, w, h in boxes]

    def _match_single(self, normed):
        """Vectorized matching against all 10 compiled templates using MSE."""
        if self.templates_3d is None:
            self._compile_templates()
            if self.templates_3d is None:
                return None, 0.0
            
        diffs = np.mean(
            np.square(self.templates_3d - normed.astype(np.float32)), 
            axis=(1, 2)
        )
        
        best_idx = int(np.argmin(diffs))
        best_digit = int(self.template_digits[best_idx])
        max_diff = 255.0 ** 2
        confidence = 1.0 - (diffs[best_idx] / max_diff)
        
        return best_digit, float(confidence)
    
    def _compile_templates(self):
        """
        Converts working templates into a single np.array for O(1) matching.
        """
        ids = []
        compiled_list = []
        for d in range(10):
            template = self._working_templates[d]
            if template is not None:
                compiled_list.append(template)
                ids.append(d)
            
        self.templates_3d = np.stack(compiled_list) if compiled_list else None
        self.template_digits = np.array(ids, dtype=np.int8) if ids else None
