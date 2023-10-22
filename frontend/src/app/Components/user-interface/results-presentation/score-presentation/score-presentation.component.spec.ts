import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ScorePresentationComponent } from './score-presentation.component';

describe('ScorePresentationComponent', () => {
  let component: ScorePresentationComponent;
  let fixture: ComponentFixture<ScorePresentationComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ ScorePresentationComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ScorePresentationComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
