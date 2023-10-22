import { ComponentFixture, TestBed } from '@angular/core/testing';

import { IsComparativeComponent } from './is-comparative.component';

describe('IsComparativeComponent', () => {
  let component: IsComparativeComponent;
  let fixture: ComponentFixture<IsComparativeComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ IsComparativeComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(IsComparativeComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
